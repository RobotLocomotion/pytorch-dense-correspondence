import unittest
import os

import torch
import numpy as np

import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename, reset_random_seed
from dense_correspondence.dataset.spartan_episode_reader import SpartanEpisodeReader
import dense_correspondence.loss_functions.utils as loss_utils
from dense_correspondence.network.predict import get_integral_preds_3d, get_integral_preds_2d
# key_dynam
from key_dynam.utils.torch_utils import random_sample_from_masked_image_torch

class TestIndexing(unittest.TestCase):


    def test_indexing(self, verbose=False):

        np.random.seed(0)
        torch.manual_seed(0)

        B = 2
        D = 3
        H = 480
        W = 640
        N = 150

        L = 10

        img = torch.zeros([B, D, H, W])
        uv = torch.zeros([B, 2, N], dtype=torch.long)
        valid = torch.zeros([B, N], dtype=torch.long)

        entry_list = []
        for i in range(L):
            b = np.random.randint(0, B)
            u = np.random.randint(0, W)
            v = np.random.randint(0, H)
            # n = np.random.randint(0, N)
            n = i # these must be in order . . .

            e = {'b': b,
                 'u': u,
                 'v': v,
                 'n': n}

            entry_list.append(e)

        for e in entry_list:
            b = e['b']
            u = e['u']
            v = e['v']
            n = e['n']

            valid[b, n] = 1
            rand_descriptor = torch.rand([D])
            img[b, :, v, u] = rand_descriptor
            uv[b, :, n] = torch.tensor([u, v], dtype=uv.dtype)

            e['descriptor'] = rand_descriptor.clone()


        # [B, D, N]
        des = pdc_utils.index_into_batch_image_tensor(img, uv, verbose=False)
        if verbose:
            print("des.shape", des.shape)

        des_valid_dict = pdc_utils.extract_valid_descriptors(des, valid, verbose=False)
        if verbose:
            print("des_valid_dict['des'].shape", des_valid_dict['des'].shape)

        for i,e in enumerate(entry_list):
            b = e['b']
            u = e['u']
            v = e['v']
            n = e['n']
            descriptor_actual = e['descriptor']

            # shape [B, D, N]
            tmp = des[b, :, n]
            allclose = torch.allclose(tmp, descriptor_actual)
            if not allclose:
                print("tmp", tmp)
                print("e['descriptor']", e['descriptor'])
                print("b", b)
                print("n", n)
                print("u", u)
                print("v", v)

            self.assertTrue(allclose, "descriptor in des[b,:,n] doesn't match")

            passed = False
            counter = 0
            tmp2 = des_valid_dict['des'][n, :]

            for e2 in entry_list:
                if torch.allclose(tmp2, e2['descriptor']):
                    counter += 1

            self.assertEqual(counter, 1, "Didn't find descriptor in des_valid_dict['des']")


        if verbose:
            print("des_valid_dict['n_idx']", des_valid_dict['n_idx'])
            print("des_valid_dict['des']\n", des_valid_dict['des'])

            for e in entry_list:
                print("descriptor:", e['descriptor'])

        for i in range(des_valid_dict['n_idx'].numel()):
            n = des_valid_dict['n_idx'][i].item()
            e = entry_list[n]

            tmp = des_valid_dict['des'][i, :]
            descriptor_actual = e['descriptor']
            allclose = torch.allclose(tmp, descriptor_actual)
            if not allclose:
                print("i", i)
                print("n", n)
                print("tmp", tmp)
                print("descriptor", descriptor_actual)

            self.assertTrue(allclose, "des_valid descriptor doesn't match actual one")


    def test_find_pixelwise_extreme(self, verbose=False):

        np.random.seed(0)
        torch.manual_seed(0)

        B = 2
        D = 3
        H = 480
        W = 640
        N = 150

        L = 10

        img = torch.zeros([B, D, H, W])
        norm_diff_max = torch.zeros([B, N, H, W])
        norm_diff_min = torch.zeros_like(norm_diff_max)

        entries = []
        for b in range(B):
            batch_entry_list = []
            for n in range(N):
                u = np.random.randint(0, W)
                v = np.random.randint(0, H)

                max_value = np.random.rand()
                min_value = -np.random.rand()

                norm_diff_max[b, n, v, u] = max_value
                norm_diff_min[b, n, v, u] = min_value

                if b == 0 and n == 0 and verbose:
                    print("this should only print once")
                    print('u:', u)
                    print('v:', v)
                    print('max_value', max_value)
                    print("min_value:", min_value)
                    print('\n')

                e = {'b': b,
                     'n': n,
                     'u': u,
                     'v': v,
                     'max_value': max_value,
                     'min_value': min_value}

                batch_entry_list.append(e)

            entries.append(batch_entry_list)

        max_dict = pdc_utils.find_pixelwise_extreme(norm_diff_max, 'max', verbose=verbose)
        values_max = max_dict['values']
        indices_max = max_dict['indices']

        min_dict = pdc_utils.find_pixelwise_extreme(norm_diff_min, 'min', verbose=verbose)
        values_min = min_dict['values']
        indices_min = max_dict['indices']

        for b in range(B):
            for n in range(N):
                e = entries[b][n]

                u = e['u']
                v = e['v']

                uv_max = indices_max[b, n, :]
                uv_max_gt = torch.tensor([u, v], dtype=uv_max.dtype)

                uv_min = indices_min[b, n, :]
                uv_min_gt = torch.tensor([u, v], dtype=uv_min.dtype)

                max_value_gt = e['max_value']
                min_value_gt = e['min_value']
                max_value = values_max[b,n].item()
                min_value = values_min[b,n].item()

                passed_max = torch.allclose(uv_max, uv_max_gt) and np.allclose(max_value, max_value_gt)
                if not passed_max:
                    print("uv_max", uv_max)
                    print("uv_max_gt", uv_max_gt)
                    print("max_value_gt:", max_value_gt)
                    print("max_value", max_value)

                self.assertTrue(passed_max, "incorrect max indices found")

                # print("min_value_gt:", min_value_gt)
                # print("min_value", min_value)
                passed_min = torch.allclose(uv_min, uv_min_gt) and np.allclose(min_value, min_value_gt)
                if not passed_min:
                    print("uv_min", uv_min)
                    print("uv_min_gt", uv_min_gt)
                    print("min_value_gt:", min_value_gt)
                    print("min_value", min_value)

    def test_norm_diff(self, verbose=True):

        B = 2
        D = 3
        H = 480
        W = 640
        N = 150
        L = 10

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        des_img_b = torch.zeros([B, D, H, W]).to(device)
        uv = torch.zeros([B, 2, N], dtype=torch.long).to(device)

        batch_des_a = pdc_utils.index_into_batch_image_tensor(des_img_b,
                                                              uv).permute([0,2,1])

        expand_batch_des_a = pdc_utils.expand_descriptor_batch(batch_des_a, H, W)
        expand_des_img_b = pdc_utils.expand_image_batch(des_img_b, N)

        if verbose:
            print("expand_batch_des_a.shape", expand_batch_des_a.shape)
            print("expand_des_img_b.shape", expand_des_img_b.shape)


        # # [B, N, D, 1, 1]
        # batch_des_a = batch_des_a.permute([0,2,1]).unsqueeze(-1).unsqueeze(-1)
        #
        # if verbose:
        #     print('batch_des_a.shape', batch_des_a.shape)
        #
        # # [B, N, D, H, W]
        # batch_des_a_expand = batch_des_a.expand(*[-1, -1, -1, H, W])
        # if verbose:
        #     print("batch_des_a_expand.shape", batch_des_a_expand.shape)
        #
        # # [B, 1, D, H, W]
        # des_img_b_expand = des_img_b.unsqueeze(1)
        # if verbose:
        #     print("des_img_b_expand.shape", des_img_b_expand.shape)
        #
        # # [B, N, D, H, W]
        # des_img_b_expand = des_img_b_expand.expand(*[-1, N, -1, -1, -1])
        # if verbose:
        #     print("des_img_b_expand.shape", des_img_b_expand.shape)


        norm_diff = (expand_batch_des_a - expand_des_img_b).norm(dim=2)

        if verbose:
            print("norm_diff.shape", norm_diff.shape)


    def test_extract_valid(self, verbose=False):
        np.random.seed(0)
        torch.manual_seed(0)

        B = 2
        N = 10
        H = 48
        W = 64
        L = 5

        x = torch.zeros([B,N,H,W])
        valid = torch.zeros([B,N], dtype=torch.long)

        entry_list = []
        for i in range(L):
            b = np.random.randint(0, B)
            # n = np.random.randint(0, N)
            n = i
            val = np.random.rand() * torch.ones([H, W])

            # n = np.random.randint(0, N)
            # n = i  # these must be in order . . .

            x[b, n, :, :] = val
            valid[b, n] = 1

            e = {'b': b,
                 'val': val,
                 'n': n}

            entry_list.append(e)

        x_valid = pdc_utils.extract_valid(x, valid)

        if verbose:
            print("x_valid.shape", x_valid.shape)

        M = x_valid.shape[0]
        for m in range(M):
            val = x_valid[m, :, :]

            counter = 0
            for e in entry_list:
                if torch.allclose(val, e['val']):
                    counter += 1
                    print("\n")
                    print("m", m)
                    print("b", e['b'])
                    print("n", e['n'])


            self.assertTrue(counter, 1)


    def test_compute_descriptor_heatmap(self, verbose=True):
        """
        This is a very simple test. Just does dimension checking.
        See jupyter notebook for more extensive test
        :param verbose:
        :type verbose:
        :return:
        :rtype:
        """

        np.random.seed(0)
        torch.manual_seed(0)

        B = 2
        N = 10
        H = 48
        W = 64
        D = 3


        img = torch.zeros([B, D, H, W])
        sigma = 1
        des = torch.zeros([B, N, D])
        type = 'exp'

        heatmap = loss_utils.compute_heatmap_from_descriptors(des,
                                                              img,
                                                              sigma,
                                                              type)

        if verbose:
            print("heatmap.shape", heatmap.shape)

        self.assertTrue(heatmap.shape[0], B)
        self.assertTrue(heatmap.shape[1], N)
        self.assertTrue(heatmap.shape[2], H)
        self.assertTrue(heatmap.shape[3], W)


    def test_heatmap_integral_pred_2d(self, verbose=False):

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'

        np.random.seed(0)
        torch.manual_seed(0)

        H = 480
        W = 640
        N = 5

        sigma = 1.0
        types = ['exp', 'softmax']

        L = 10
        tol = 0.03 # tolerance in pixels

        uv_input = torch.zeros([N, 2], dtype=torch.long).to(device)

        for i in range(N):
            u = np.random.randint(0, W)
            v = np.random.randint(0, H)
            uv_input[i, 0] = u
            uv_input[i, 1] = v


        heatmaps = dict()
        preds = dict()
        for type in types:
            heatmaps[type] = loss_utils.create_heatmap(uv_input, H=H, W=W, sigma=sigma, type=type)
            preds[type] = get_integral_preds_2d(heatmaps[type])

            print("preds.shape", preds[type].shape)


        for type in types:

            # size N
            diff = uv_input.type(torch.float) - preds[type]

            max_diff = torch.max(diff)
            self.assertTrue(max_diff < tol)

            if verbose:
                print("\n")
                print("type:", type)
                print("max_diff", max_diff)

    def test_heatmap_integral_pred_3d(self, verbose=True):

        reset_random_seed(SEED=1)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load Data
        dataset_processed_dir = os.path.join(getDenseCorrespondenceSourceDir(), 'dense_correspondence/test/data/2018-04-16-14-25-19/processed')

        config_file = os.path.join(getDenseCorrespondenceSourceDir(),
                                   'config/dense_correspondence/global/drake_sim_dynamic.yaml')
        config = getDictFromYamlFilename(config_file)
        episode = SpartanEpisodeReader(config, dataset_processed_dir)

        idx = episode.indices[0]
        camera_name = episode.camera_names[0]
        data = episode.get_image_data(camera_name, idx)


        depth_img_rendered = torch.Tensor(data['depth_int16'].astype(np.int32)).to(device)
        depth_img_raw = torch.Tensor(episode.get_raw_depth_image_int16(camera_name, idx)).to(device)

        # run the test for both types of depth images
        for depth_img, depth_img_type in zip([depth_img_rendered, depth_img_raw], ["rendered", "raw"]):
            reset_random_seed(SEED=1)

            sigma = 1.0
            types = ['exp']

            tol = 0.03# tolerance in pixels
            z_tol = 10 # tolerance in mm

            mask = torch.Tensor(data['mask'])

            N = 5
            H, W = mask.shape

            # [N, H, W]
            depth_img_expand = depth_img.unsqueeze(0).expand([N, -1, -1])

            # [N, 2] in u,v ordering
            uv_input = random_sample_from_masked_image_torch(mask, N).to(device)
            valid_idx = depth_img[uv_input[:,1], uv_input[:, 0]] > 0
            uv_input = uv_input[valid_idx, :]


            heatmaps = dict()
            preds = dict()
            for type in types:
                heatmaps[type] = loss_utils.create_heatmap(uv_input, H=H, W=W, sigma=sigma, type=type).to(device)
                preds[type] = get_integral_preds_3d(heatmaps[type], depth_img_expand)

            for type in types:

                # size N
                diff_uv = torch.abs(uv_input.type(torch.float) - preds[type]['uv'])

                max_diff = torch.max(diff_uv)
                self.assertTrue(max_diff < tol)

                depth_values = depth_img[uv_input[:, 1], uv_input[:, 0]]

                diff_z = torch.abs(depth_values - preds[type]['z'])
                max_diff_z = torch.max(diff_z).item()


                self.assertTrue(max_diff_z < z_tol)

                if verbose:
                    print("\n")
                    print("depth_img_type:", depth_img_type)
                    print("type:", type)
                    print("max_diff", max_diff)
                    print("max_diff_z", max_diff_z)




if __name__ == "__main__":
    unittest.main()