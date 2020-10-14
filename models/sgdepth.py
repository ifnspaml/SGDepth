import torch
import torch.nn as nn

from . import networks
from . import layers


class SGDepthCommon(nn.Module):
    def __init__(self, num_layers, split_pos, grad_scales=(0.9, 0.1), pretrained=False):
        super().__init__()

        self.encoder = networks.ResnetEncoder(num_layers, pretrained)
        self.num_layers = num_layers  # This information is needed in the train loop for the sequential training

        # Number of channels for the skip connections and internal connections
        # of the decoder network, ordered from input to output
        self.shape_enc = tuple(reversed(self.encoder.num_ch_enc))
        self.shape_dec = (256, 128, 64, 32, 16)

        self.decoder = networks.PartialDecoder.gen_head(self.shape_dec, self.shape_enc, split_pos)
        self.split = layers.ScaledSplit(*grad_scales)

    def set_gradient_scales(self, depth, segmentation):
        self.split.set_scales(depth, segmentation)

    def get_gradient_scales(self):
        return self.split.get_scales()

    def forward(self, x):
        # The encoder produces outputs in the order
        # (highest res, second highest res, …, lowest res)
        x = self.encoder(x)

        # The decoder expects it's inputs in the order they are
        # used. E.g. (lowest res, second lowest res, …, highest res)
        x = tuple(reversed(x))

        # Replace some elements in the x tuple by decoded
        # tensors and leave others as-is
        x = self.decoder(*x) # CHANGE ME BACK TO THIS

        # Setup gradient scaling in the backward pass
        x = self.split(*x)

        # Experimental Idea: We want the decoder layer to be trained, so pass x to the decoder AFTER x was passed
        # to self.split which scales all gradients to 0 (if grad_scales are 0)
        # x = (self.decoder(*x[0]), ) + (self.decoder(*x[1]), ) + (self.decoder(*x[2]), )

        return x

    #def get_last_shared_layer(self):
    #    return self.encoder.encoder.layer4


class SGDepthDepth(nn.Module):
    def __init__(self, common, resolutions=1):
        super().__init__()

        self.resolutions = resolutions

        self.decoder = networks.PartialDecoder.gen_tail(common.decoder)
        self.multires = networks.MultiResDepth(self.decoder.chs_x()[-resolutions:])

    def forward(self, *x):
        x = self.decoder(*x)
        x = self.multires(*x[-self.resolutions:])
        return x


class SGDepthSeg(nn.Module):
    def __init__(self, common):
        super().__init__()

        self.decoder = networks.PartialDecoder.gen_tail(common.decoder)
        self.multires = networks.MultiResSegmentation(self.decoder.chs_x()[-1:])
        self.nl = nn.Softmax2d()

    def forward(self, *x):
        x = self.decoder(*x)
        x = self.multires(*x[-1:])
        x_lin = x[-1]

        return x_lin


class SGDepthPose(nn.Module):
    def __init__(self, num_layers, pretrained=False):
        super().__init__()

        self.encoder = networks.ResnetEncoder(
            num_layers, pretrained, num_input_images=2
        )

        self.decoder = networks.PoseDecoder(self.encoder.num_ch_enc[-1])

    def _transformation_from_axisangle(self, axisangle):
        n, h, w = axisangle.shape[:3]

        angles = axisangle.norm(dim=3)
        axes = axisangle / (angles.unsqueeze(-1) + 1e-7)

        # Implement the matrix from [1] with an additional identity fourth dimension
        # [1]: https://en.wikipedia.org/wiki/Transformation_matrix#Rotation_2

        angles_cos = angles.cos()
        angles_sin = angles.sin()

        res = torch.zeros(n, h, w, 4, 4, device=axisangle.device)
        res[:,:,:,:3,:3] = (1 - angles_cos.view(n,h,w,1,1)) * (axes.unsqueeze(-2) * axes.unsqueeze(-1))

        res[:,:,:,0,0] += angles_cos
        res[:,:,:,1,1] += angles_cos
        res[:,:,:,2,2] += angles_cos

        sl = axes[:,:,:,0] * angles_sin
        sm = axes[:,:,:,1] * angles_sin
        sn = axes[:,:,:,2] * angles_sin

        res[:,:,:,0,1] -= sn
        res[:,:,:,1,0] += sn

        res[:,:,:,1,2] -= sl
        res[:,:,:,2,1] += sl

        res[:,:,:,2,0] -= sm
        res[:,:,:,0,2] += sm

        res[:,:,:,3,3] = 1.0

        return res

    def _transformation_from_translation(self, translation):
        n, h, w = translation.shape[:3]

        # Implement the matrix from [1] with an additional dimension
        # [1]: https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations

        res = torch.zeros(n, h, w, 4, 4, device=translation.device)
        res[:,:,:,:3,3] = translation
        res[:,:,:,0,0] = 1.0
        res[:,:,:,1,1] = 1.0
        res[:,:,:,2,2] = 1.0
        res[:,:,:,3,3] = 1.0

        return res

    def forward(self, x, invert):
        x = self.encoder(x)
        x = x[-1]  # take only the feature map of the last layer ...

        x_axisangle, x_translation = self.decoder(x)  # ... and pass it through the decoder

        x_rotation = self._transformation_from_axisangle(x_axisangle)

        if not invert:
            x_translation = self._transformation_from_translation(x_translation)

            return x_translation @ x_rotation

        else:
            x_rotation = x_rotation.transpose(3, 4)
            x_translation = -x_translation

            x_translation = self._transformation_from_translation(x_translation)

            return x_rotation @ x_translation


class SGDepth(nn.Module):
    KEY_FRAME_CUR = ('color_aug', 0, 0)
    KEY_FRAME_PREV = ('color_aug', -1, 0)
    KEY_FRAME_NEXT = ('color_aug', 1, 0)

    def __init__(self, split_pos=1, num_layers=18, grad_scale_depth=0.95, grad_scale_seg=0.05,
                 weights_init='pretrained', resolutions_depth=1, num_layers_pose=18):

        super().__init__()

        # sgdepth allowed for five possible split positions.
        # The PartialDecoder developed as part of sgdepth
        # is a bit more flexible and allows splits to be
        # placed in between sgdepths splits.
        # As this class is meant to maximize compatibility
        # with sgdepth the line below translates between
        # the split position definitions.
        split_pos = max((2 * split_pos) - 1, 0)

        # The Depth and the Segmentation Network have a common (=shared)
        # Encoder ("Feature Extractor")
        self.common = SGDepthCommon(
            num_layers, split_pos, (grad_scale_depth, grad_scale_seg),
            weights_init == 'pretrained'
        )

        # While Depth and Seg Network have a shared Encoder,
        # each one has it's own Decoder
        self.depth = SGDepthDepth(self.common, resolutions_depth)
        self.seg = SGDepthSeg(self.common)

        # The Pose network has it's own Encoder ("Feature Extractor") and Decoder
        self.pose = SGDepthPose(
            num_layers_pose,
            weights_init == 'pretrained'
        )

    def _batch_pack(self, group):
        # Concatenate a list of tensors and remember how
        # to tear them apart again

        group = tuple(group)

        dims = tuple(b.shape[0] for b in group)  # dims = (DEFAULT_DEPTH_BATCH_SIZE, DEFAULT_SEG_BATCH_SIZE)
        group = torch.cat(group, dim=0)  # concatenate along the first axis, so along the batch axis

        return dims, group

    def _multi_batch_unpack(self, dims, *xs):
        xs = tuple(
            tuple(x.split(dims))
            for x in xs
        )

        # xs, as of now, is indexed like this:
        # xs[ENCODER_LAYER][DATASET_IDX], the lines below swap
        # this around to xs[DATASET_IDX][ENCODER_LAYER], so that
        # xs[DATASET_IDX] can be fed into the decoders.
        xs = tuple(zip(*xs))

        return xs

    def _check_purposes(self, dataset, purpose):
        # mytransforms.AddKeyValue is used in the loaders
        # to give each image a tuple of 'purposes'.
        # As of now these purposes can be 'depth' and 'segmentation'.
        # The torch DataLoader collates these per-image purposes
        # into list of them for each batch.
        # Check all purposes in this collated list for the requested
        # purpose (if you did not do anything wonky all purposes in a
        # batch should be equal),

        for purpose_field in dataset['purposes']:
            if purpose_field[0] == purpose:
                return True

    def set_gradient_scales(self, depth, segmentation):
        self.common.set_gradient_scales(depth, segmentation)

    def get_gradient_scales(self):
        return self.common.get_gradient_scales()

    def forward(self, batch):
        # Stitch together all current input frames
        # in the input group. So that batch normalization
        # in the encoder is done over all datasets/domains.
        dims, x = self._batch_pack(
            dataset[self.KEY_FRAME_CUR]
            for dataset in batch
        )

        # Feed the stitched-together input tensor through
        # the common network part and generate two output
        # tuples that look exactly the same in the forward
        # pass, but scale gradients differently in the backward pass.
        x_depth, x_seg = self.common(x)

        # Cut the stitched-together tensors along the
        # dataset boundaries so further processing can
        # be performed on a per-dataset basis.
        # x[DATASET_IDX][ENCODER_LAYER]
        x_depth = self._multi_batch_unpack(dims, *x_depth)
        x_seg = self._multi_batch_unpack(dims, *x_seg)

        outputs = list(dict() for _ in batch)

        # All the way back in the loaders each dataset is assigned one or more 'purposes'.
        # For datasets with the 'depth' purpose set the outputs[DATASET_IDX] dict will be
        # populated with depth outputs.
        # Datasets with the 'segmentation' purpose are handled accordingly.
        # If the pose outputs are populated is dependant upon the presence of
        # ('color_aug', -1, 0)/('color_aug', 1, 0) keys in the Dataset.
        for idx, dataset in enumerate(batch):
            if self._check_purposes(dataset, 'depth'):
                x = x_depth[idx]
                x = self.depth(*x)
                x = reversed(x)

                for res, disp in enumerate(x):
                    outputs[idx]['disp', res] = disp

            if self._check_purposes(dataset, 'segmentation'):
                x = x_seg[idx]
                x = self.seg(*x)

                outputs[idx]['segmentation_logits', 0] = x

            if self.KEY_FRAME_PREV in dataset:
                frame_prev = dataset[self.KEY_FRAME_PREV]
                frame_cur = dataset[self.KEY_FRAME_CUR]

                # Concatenating joins the previous and the current frame
                # tensors along the first axis/dimension,
                # which is the axis for the color channel
                frame_prev_cur = torch.cat((frame_prev, frame_cur), dim=1)

                outputs[idx]['cam_T_cam', 0, -1] = self.pose(frame_prev_cur, invert=True)

            if self.KEY_FRAME_NEXT in dataset:
                frame_cur = dataset[self.KEY_FRAME_CUR]
                frame_next = dataset[self.KEY_FRAME_NEXT]

                frame_cur_next = torch.cat((frame_cur, frame_next), 1)
                outputs[idx]['cam_T_cam', 0, 1] = self.pose(frame_cur_next, invert=False)

        return tuple(outputs)
