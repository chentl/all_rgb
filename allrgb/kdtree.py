from allrgb.lab import *
from allrgb.cache import CacheStorage, CacheLoadError


_cache_storage = CacheStorage('kdtree')


class RGBKdNode:
    """ Class for nodes in RGB k-D tree. """

    def __init__(self, n, root, nodes, count):
        """
        Create a RGB k-D node.

        :param n: index of current node (0~7).
        :param root: point to parent node.
        :param nodes: list of children nodes.
        :param count: number of available colors left in the subtree from this node.
        """
        self.n = n
        self.root = root
        self.nodes = nodes
        self.count = count
        self.lab = None


class RGBKdTree:
    """ Class for RGB k-D tree object. """

    def __init__(self, bits=8):
        """
        Create a new RGB k-D tree.

        :param bits: (optional) bit depth of colors. Currently the only allowed value is 8.
        """
        assert bits in [8]
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
        """ return (r, g, b) tuple of given node """

        path_int = []
        while node.n is not None:
            path_int.append(node.n)
            node = node.root

        path_int.reverse()
        path_int.extend([0] * (8 - len(path_int)))

        bin_path_int = [bin(p)[2:].rjust(3, '0') for p in path_int]
        return tuple([int(''.join(b), 2) for b in zip(*bin_path_int)])

    def _rgb_to_path_int(self, r, g, b):
        """ return path list to the given (r, g, b) color in tree"""

        return [int(a + b + c, base=2) for a, b, c in zip(bin(r)[2:].rjust(8, '0'), bin(g)[2:].rjust(8, '0'),
                                                          bin(b)[2:].rjust(8, '0'))][:self.bits]

    def pop_nearest_neighbor(self, r_ref, g_ref, b_ref, slow=False):
        """
        Pop the nearest color from k-D tree and return.

        :param r_ref: R component of target color.
        :param g_ref: G component of target color.
        :param b_ref: B component of target color.
        :param slow: (optional) Use slow mode. Set this to True will use CIE94 to
         calculate color differences instead of faster CIE76.
        :return: nearest color from k-D tree as a (r, g, b) tuple.
        """

        rgb_to_lab = self._rgb2lab_map.get
        path_int_ref = self._rgb_to_path_int(r_ref, g_ref, b_ref)
        lab_ref = rgb_to_lab(r_ref, g_ref, b_ref)
        delta_e = delta_e_94 if slow else delta_e_76

        # Walk down along the tree to find the nearest available color in tree.
        path_list, current_node = [], self.root
        for p_ref in path_int_ref:
            if current_node.nodes[p_ref].count > 0:
                path_list.append(p_ref)
                current_node = current_node.nodes[p_ref]
            else:
                available_nodes = [node for node in current_node.nodes if node.count > 0]
                min_node = min(available_nodes, key=lambda node: delta_e(node.lab, lab_ref))
                path_list.append(min_node.n)
                current_node = min_node
            current_node.count = current_node.count - 1

        bin_path_int = [bin(p)[2:].rjust(3, '0') for p in path_list]
        r_new, g_new, b_new = tuple([int(''.join(b), 2) for b in zip(*bin_path_int)])

        return r_new, g_new, b_new

