from allrgb.lab import *
from allrgb.cache import CacheStorage, CacheLoadError


_cache_storage = CacheStorage('kdtree')


class RGBKdNode:
    def __init__(self, n, root, nodes, count):
        self.n = n
        self.root = root
        self.nodes = nodes
        self.count = count
        # self.rgb = None
        self.lab = None
        # self.left = None
        # self.right = None


class RGBKdTree:
    def __init__(self, bits=8):
        assert bits in range(1, 9)
        self.bits = bits

        cache_tag = ('RGBKdTree', str(bits), 'Rgb2LabMap')
        try:
            self._rgb2lab_map = _cache_storage.load_cache(cache_tag)
        except CacheLoadError:
            self._rgb2lab_map = Rgb2LabMap(bits=bits)
            _cache_storage.write_cache(cache_tag, self._rgb2lab_map)

        cache_tag = ('RGBKdTree', str(bits), 'Tree')
        try:
            self.root = _cache_storage.load_cache(cache_tag)
        except CacheLoadError:
            self.root = RGBKdNode(None, None, None, (2 ** bits) ** 3)
            roots = [self.root]
            rgb_to_lab = self._rgb2lab_map.get
            for level in range(bits):
                new_roots = []
                for root in roots:
                    nodes = [RGBKdNode(n, root, None, (2 ** (bits - level - 1)) ** 3) for n in range(8)]
                    if level == (bits - 1):
                        for i in range(8):
                            rgb = self._node_to_rgb(nodes[i])
                            assert nodes[i].lab is None
                            nodes[i].lab = rgb_to_lab(*rgb)
                    root.nodes = nodes
                    new_roots.extend(nodes)
                roots = new_roots
            while roots[0].root is not None:
                new_roots = list(set([n.root for n in roots]))
                for root in new_roots:
                    assert root.lab is None
                    root.lab = (sum([node.lab[0] for node in root.nodes]) / 8.0,
                                sum([node.lab[1] for node in root.nodes]) / 8.0,
                                sum([node.lab[2] for node in root.nodes]) / 8.0)
                roots = new_roots
            _cache_storage.write_cache(cache_tag, self.root)

    @staticmethod
    def _node_to_rgb(node):
        path_int = []
        while node.n is not None:
            path_int.append(node.n)
            node = node.root

        path_int.reverse()
        path_int.extend([0] * (8 - len(path_int)))

        bin_path_int = [bin(p)[2:].rjust(3, '0') for p in path_int]
        return tuple([int(''.join(b), 2) for b in zip(*bin_path_int)])

    def _rgb_to_path_int(self, r, g, b):
        return [int(a + b + c, base=2) for a, b, c in zip(bin(r)[2:].rjust(8, '0'), bin(g)[2:].rjust(8, '0'),
                                                          bin(b)[2:].rjust(8, '0'))][:self.bits]

    def pop_nearest_neighbor(self, r_ref, g_ref, b_ref):
        rgb_to_lab = self._rgb2lab_map.get
        path_int_ref = self._rgb_to_path_int(r_ref, g_ref, b_ref)
        # print('[RGBKdTree] [pop] path_ref_int:', path_int_ref)
        lab_ref = rgb_to_lab(r_ref, g_ref, b_ref)
        # print('[RGBKdTree] [pop] lab_ref:', lab_ref)

        path_list, current_node = [], self.root
        for p_ref in path_int_ref:
            if current_node.nodes[p_ref].count > 0:
                path_list.append(p_ref)
                current_node = current_node.nodes[p_ref]
            else:
                available_nodes = [node for node in current_node.nodes if node.count > 0]
                min_node = min(available_nodes, key=lambda node: delta_e_76(node.lab, lab_ref))
                path_list.append(min_node.n)
                current_node = min_node
            current_node.count = current_node.count - 1

        bin_path_int = [bin(p)[2:].rjust(3, '0') for p in path_list]
        r_new, g_new, b_new = tuple([int(''.join(b), 2) for b in zip(*bin_path_int)])

        # while current_node.root is not None:
        #     current_node.count = current_node.count - 1
        #     current_node = current_node.root
        # current_node.count = current_node.count - 1

        return r_new, g_new, b_new

