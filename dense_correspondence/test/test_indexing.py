import unittest

import torch
import numpy as np

import dense_correspondence_manipulation.utils.utils as pdc_utils
import dense_correspondence.loss_functions.utils as loss_utils

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


    def test_find_pixelwise_extreme(self, verbose=True):

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

        np.random.seed(0)
        torch.manual_seed(0)

        B = 2
        N = 10
        H = 48
        W = 64
        D = 3
        L = 5


        img = torch.zeros([B, H, W, D])
        sigma = 1
        des = torch.zeros([B, N, D])
        type = 'exp'

        heatmap = loss_utils.compute_heatmap_from_descriptors(des,
                                                              img,
                                                              sigma,
                                                              type)

        if verbose:
            print("heatmap.shape", heatmap.shape)









if __name__ == "__main__":
    unittest.main()