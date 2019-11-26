import unittest

import torch
import numpy as np

import dense_correspondence_manipulation.utils.utils as pdc_utils

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
        norm_diff = torch.zeros([B, N, H, W])

        entries = []
        for b in range(B):
            batch_entry_list = []
            for n in range(N):
                u = np.random.randint(0, W)
                v = np.random.randint(0, H)

                norm_diff[b, n, v, u] = 1.0

                if b == 0 and n == 0 and verbose:
                    print("this should only print once")
                    print('u:', u)
                    print('v:', v)
                    print('\n')

                e = {'b': b,
                     'n': n,
                     'u': u,
                     'v': v}

                batch_entry_list.append(e)

            entries.append(batch_entry_list)


        indices_max = pdc_utils.find_pixelwise_extreme(norm_diff, 'max', verbose=verbose)

        for b in range(B):
            for n in range(N):
                e = entries[b][n]

                u = e['u']
                v = e['v']


                uv_max = indices_max[b, n, :]
                uv = torch.tensor([u, v], dtype=uv_max.dtype)

                passed_max = torch.allclose(uv, uv_max)
                if not passed_max:
                    print("uv", uv)
                    print("uv_max", uv_max)


                self.assertTrue(passed_max, "incorrect max indices found")






if __name__ == "__main__":
    unittest.main()