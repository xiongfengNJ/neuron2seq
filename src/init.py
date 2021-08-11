# coding=utf-8
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import neuro_morpho_toolbox as nmt
import random
import os

try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder  # Scikit-Learn < 0.20
try:
    from sklearn.impute import SimpleImputer  # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

sys.setrecursionlimit(3000)


class TreeNode:
    def __init__(self, key, val, left=None, right=None, parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        self.balanceFactor = 0

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def getRootVal(self):
        return self.payload


class BinarySearchTree:
    def __init__(self, keep=True, stem=None):  # stem=[] //in class neuron
        self.root = None
        self.size = 0
        self.stems = []

        if stem:
            # print('keep: ',keep)
            self.stem2seq(stem, keep)

    def delete_grow_nodes(self, root):
        if not root:
            return
        else:
            left = root.leftChild
            right = root.rightChild
            self.__delete(left)
            self.__delete(right)
        return

    def __delete(self, root):
        if not root:
            return
        left = root.leftChild
        right = root.rightChild

        if (left and right):
            self.__delete(left)
            self.__delete(right)
            return
        elif (left or right):

            if left:
                if root.parent.leftChild == root:
                    root.parent.leftChild = left
                    left.parent = root.parent
                    left.payload = left.payload + root.payload
                elif root.parent.rightChild == root:
                    root.parent.rightChild = left
                    left.parent = root.parent
                    left.payload = left.payload + root.payload
                # self.__delete(root.parent)
                self.__delete(left)
                self.__delete(right)

            elif right:
                if root.parent.leftChild == root:
                    root.parent.leftChild = right
                    right.parent = root.parent
                    right.payload = right.payload + root.payload
                elif root.parent.rightChild == root:
                    root.parent.rightChild = right
                    right.parent = root.parent
                    right.payload = right.payload + root.payload
                # self.__delete(root.parent)
                self.__delete(left)
                self.__delete(right)

            return

    def stem2seq(self, stem, keep):
        # print('keep_grow:', keep_grow)
        for seq in stem:
            if not self.root:
                self.root = TreeNode(key=seq[-1][0], val=seq[-1][1], )
                self.root.parent = None
            cur = self.root
            for i in range(len(seq) - 1, -1, -1):
                node = seq[i]
                #         print(node)
                #         print('=======')
                if node[0] == cur.key:
                    continue

                elif cur.hasLeftChild():
                    if node[0] == cur.leftChild.key:
                        cur = cur.leftChild
                        continue
                else:
                    new_node = TreeNode(key=node[0], val=node[1], parent=cur)
                    cur.leftChild = new_node
                    new_node.parent = cur
                    cur = cur.leftChild
                    continue

                if cur.hasRightChild():
                    if node[0] == cur.rightChild.key:
                        cur = cur.rightChild
                        continue
                else:
                    new_node = TreeNode(key=node[0], val=node[1], parent=cur)
                    cur.rightChild = new_node
                    new_node.parent = cur
                    cur = cur.rightChild
                    continue
        if not keep:
            # print('1')
            self.delete_grow_nodes(self.root)
        self.__sort_children(self.root)  # 神经元成树的过程以叶子结点数量决定遍历次序

    def __sort_children(self, root):
        if root:
            leave_l = leave(root.leftChild)
            leave_r = leave(root.rightChild)
            # len_l = path_length(root.leftChild)
            # len_r = path_length(root.rightChild)

            l_pr = leave_l
            r_pr = leave_r

            # print(l_pr,r_pr)

            if (r_pr < l_pr):
                if (r_pr != 0):
                    temp = root.leftChild
                    root.leftChild = root.rightChild
                    root.rightChild = temp
                else:
                    pass
            self.__sort_children(root.leftChild)
            self.__sort_children(root.rightChild)
        else:
            return

    def seqs2stems(self, seqs):
        k = 2
        tree_nodes = {}
        id_seqs = []
        stems = []

        for seq in seqs:
            tree_node = TreeNode(key=seq, val=k)
            tree_nodes[k] = tree_node
            id_seqs.append([k, seq])
            k = k + 1
        print(id_seqs)
        merge = False

        while (not merge):
            temp = 0
            dist = []
            for id_seq in id_seqs:
                if id_seq[1] == 't':
                    temp = temp + 1
                elif id_seq[1] == 'b':
                    temp = 0
                dist.append(temp)
            #             print('dist',dist)
            if 0 not in dist:
                merge = True
                break

            remove_list = []
            for i in range(len(dist)):
                if dist[i] == 2:
                    tree_nodes[id_seqs[i - 2][0]].leftChild = tree_nodes[id_seqs[i - 1][0]]
                    tree_nodes[id_seqs[i - 2][0]].rightChild = tree_nodes[id_seqs[i][0]]
                    #             print(id_seqs[i][0])
                    #             print(id_seqs[i-1][0])
                    #             print(id_seqs[i-2][0])
                    id_seqs[i - 2][1] = 't'
                else:
                    continue

                remove_list.append(id_seqs[i - 1])
                remove_list.append(id_seqs[i])

            #             print('remove: ',remove_list)
            for i in remove_list:
                id_seqs.remove(i)

        for i in id_seqs:
            self.stems.append(tree_nodes[i[0]])
        return self.stems

    #             print('id_seqs: ',id_seqs)

    def inorder(self):
        self._inorder(self.root)

    def _inorder(self, tree):
        if tree != None:
            self._inorder(tree.leftChild)
            print(tree.key)
            self._inorder(tree.rightChild)

    def postorder(self):
        self._postorder(self.root)

    def _postorder(self, tree):
        if tree:
            self._postorder(tree.rightChild)
            self._postorder(tree.leftChild)
            print(tree.key)

    def preorder(self):
        L = []
        self._preorder(self.root, L)
        return L

    def _preorder(self, root, L):  # 以叶子结点为依据决定先后遍历的序列
        if root:
            L.append(root.key)
            leave_l = leave(root.leftChild)
            leave_r = leave(root.rightChild)
            # len_l = path_length(root.leftChild)
            # len_r = path_length(root.rightChild)

            # l_pr = leave_l * len_l
            # r_pr = leave_r * len_r
            l_pr = leave_l
            r_pr = leave_r

            if l_pr < r_pr:  # 先遍历小的，后遍历大的
                self._preorder(root.leftChild, L)
                self._preorder(root.rightChild, L)
            else:
                self._preorder(root.rightChild, L)
                self._preorder(root.leftChild, L)
        return


d_hor = 0.5  # 节点水平距离
d_vec = 0.5  # 节点垂直距离
radius = 0.0  # 节点的半径


def leave(root):  # 递归求叶子节点个数
    if root == None:
        return 0
    elif root.leftChild == None and root.rightChild == None:
        return 1
    else:
        return (leave(root.leftChild) + leave(root.rightChild))


def path_length(root):
    if root == None:
        return 0
    return path_length(root.leftChild) + path_length(root.rightChild) + root.payload


def get_left_width(root):
    '''获得根左边宽度'''
    return get_width(root.leftChild)


def get_right_width(root):
    '''获得根右边宽度'''
    return get_width(root.rightChild)


def get_width(root):
    '''获得树的宽度'''
    if root == None:
        return 0
    return get_width(root.leftChild) + 1 + get_width(root.rightChild)


def get_height(root):
    '''获得二叉树的高度'''
    if root == None:
        return 0
    return max(get_height(root.leftChild), get_height(root.rightChild)) + 1


def get_w_h(root):
    '''返回树的宽度和高度'''
    w = get_width(root)
    h = get_height(root)
    return w, h


def draw_a_node(x, y, val, ax):
    '''画一个节点'''
    c_node = patches.Circle((x, y), radius=radius, color='green')
    ax.add_patch(c_node)
    plt.text(x, y, '%d' % val, ha='center', va='bottom', fontsize=11)


def draw_a_edge(x1, y1, x2, y2, r=radius):
    '''画一条边'''
    X0 = (x1, x1)
    Y0 = (y1, y2)
    X1 = (x2, x1)
    Y1 = (y2, y2)
    plt.plot(X0, Y0, 'k-')
    plt.plot(X1, Y1, 'k-')
    return


def create_win(root):
    '''创建窗口'''
    WEIGHT, HEIGHT = get_w_h(root)
    WEIGHT = (WEIGHT + 2) * d_hor
    HEIGHT = (HEIGHT + 2) * d_vec

    fig = plt.figure(figsize=(np.sum(WEIGHT), np.max(HEIGHT)))
    ax = fig.add_subplot(111)
    plt.xlim(0, (WEIGHT))
    plt.ylim(0, (HEIGHT))

    x = (get_left_width(root) + 1) * d_hor  # x, y 是第一个要绘制的节点坐标，由其左子树宽度决定
    y = HEIGHT - d_vec

    return fig, ax, x, y


def print_tree_by_inorder(root, x, y, ax):
    '''通过中序遍历打印二叉树'''
    if root == None:
        return

    draw_a_node(x, y, root.key, ax)
    #     draw_a_node(x, y, root.payload, ax)
    lx = rx = 0
    ly = ry = y - d_vec
    if root.leftChild != None:
        lx = x - d_hor * (get_right_width(root.leftChild) + 1)  # x-左子树的右边宽度,左孩子的x坐标
        draw_a_edge(x, y, lx, ly, radius)
    else:
        X0 = (x, x)
        Y0 = (y - d_vec, y)
        plt.plot(X0, Y0, 'k-')

    if root.rightChild != None:
        rx = x + d_hor * (get_left_width(root.rightChild) + 1)  # x-右子树的左边宽度，右孩子的x坐标
        draw_a_edge(x, y, rx, ry, radius)
    else:
        X0 = (x, x)
        Y0 = (y - d_vec, y)
        plt.plot(X0, Y0, 'k-')

    # 递归打印
    #     leave_l = leave(root.leftChild)
    #     leave_r = leave(root.rightChild)
    #     len_l = path_length(root.leftChild)
    #     len_r = path_length(root.rightChild)

    #     l_pr = leave_l * len_l
    #     r_pr = leave_r * len_r

    print_tree_by_inorder(root.leftChild, lx, ly, ax)
    print_tree_by_inorder(root.rightChild, rx, ry, ax)

    return


def show_BTree(root):
    '''可视化二叉树'''
    _, ax, x, y = create_win(root)
    print_tree_by_inorder(root, x, y, ax)
    plt.show()
    return


def preorder(tree):
    if tree != None:
        print(tree.key)
        preorder(tree.hasLeftChild())
        preorder(tree.hasRightChild())


class neuron:
    def __init__(self, neuron_path=None, dendrites_type='basal', scale_=0):
        self.path = neuron_path

        if self.path:
            n_skip = 0
            with open(self.path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line.startswith("#"):
                        n_skip += 1
                    else:
                        break
            self.n = pd.read_csv(self.path, sep=" ", index_col=0, header=None, usecols=[0, 1, 2, 3, 4, 5, 6],
                                 skiprows=n_skip, names=["##n", "type", "x", "y", "z", "r", "parent"])

            if scale_ != 0:
                # if scale_ == 0.5:
                #     self.n[['x']] = self.n[['x']] + 0.5
                #     self.n[['y']] = self.n[['y']] + 0.5
                #
                # if scale_ == 1.0:
                #     self.n[['y']] = self.n[['y']] + 0.5
                #     self.n[['z']] = self.n[['z']] + 0.5
                #
                # if scale_ == 1.5:
                #     self.n[['z']] = self.n[['z']] + 0.5
                #     self.n[['x']] = self.n[['x']] + 0.5
                #
                # if scale_ == 2:
                #     self.n[['x']] = self.n[['x']] + 0.5
                #     self.n[['y']] = self.n[['y']] + 0.5
                #     self.n[['x']] = self.n[['x']] + 0.5
                #
                # if scale_ == 2.5:
                #     self.n[['x']] = self.n[['x']] + 1
                #     self.n[['y']] = self.n[['y']] + 0.5

                print('有 loc itf ，scale = ', scale_)
                delta = np.random.normal(loc=0, scale=scale_, size=(len(self.n['x']), 3))
                self.n[['x', 'y', 'z']] = self.n[['x', 'y', 'z']] + delta
            else:
                print("scale = ", scale_)

            self.n["branch_order"] = 0

            # self.__add_branch_order()
            self.__add_NodeTypeLabel()

            self.n = self.add_region(self.n)

            self.n = self.__add_vector(self.n)  # 用于删除小分支
            self.del_small_segments()

            if dendrites_type == 'basal':
                self.dendrites_id = 3
                self.dendrites = self.n[(self.n.type == self.dendrites_id)]

            elif dendrites_type == 'apical':
                self.dendrites_id = 4
                self.dendrites = self.n[(self.n.type == self.dendrites_id)]

            elif dendrites_type == None:
                self.dendrites_id = 3
                self.dendrites = self.n[(self.n.type == 3) | (self.n.type == 4)]

            self.axon = self.n[(self.n.type == 2)]
            self.soma = self.n[self.n.type == 1]

            # self.node_type = self.__NodeTypeCon()

        return

    def get_node_region(self, point):
        p = point[['x', 'y', 'z']].copy()
        p['x'] = p['x'] / nmt.annotation.space['x']
        p['y'] = p['y'] / nmt.annotation.space['y']
        p['z'] = p['z'] / nmt.annotation.space['z']
        p = p.round(0).astype(int)
        if ((p.x.iloc[0] >= 0) & (p.x.iloc[0] < nmt.annotation.size['x']) &
                (p.y.iloc[0] >= 0) & (p.y.iloc[0] < nmt.annotation.size['y']) &
                (p.z.iloc[0] >= 0) & (p.z.iloc[0] < nmt.annotation.size['z'])
        ):
            region_id = nmt.annotation.array[p.x.iloc[0],
                                             p.y.iloc[0],
                                             p.z.iloc[0]]
            if region_id in list(nmt.bs.dict_to_selected.keys()):
                region_id = nmt.bs.dict_to_selected[region_id]
                region = nmt.bs.id_to_name(region_id)
                return region

        return 'unknow'

    def add_region(self, df):
        df['region'] = 0
        # soma = self.n[((self.n.type == 1) & (self.n.parent == -1))]
        # if soma.z.iloc[0] < (nmt.annotation.micron_size["z"] / 2):
        #     self.hemi = 1
        # else:
        #     self.hemi = 2
        # for i in df.index.tolist():
        #     point = df.loc[[i], ['x', 'y', 'z']]
        #     region = self.get_node_region(point)
        #     if (region not in ['unknow', 'fiber_tracts']):
        #         p_hemi = 1 if point.z.iloc[0] < (nmt.annotation.micron_size["z"] / 2) else 2
        #         if p_hemi == self.hemi:
        #             region = "s_" + region
        #         else:
        #             # print("region ",region)
        #             region = "c_" + region
        #     df.loc[i, 'region'] = region
        df['region'] = df.apply(lambda r: self.get_node_region(pd.DataFrame({'x': [r.x], 'y': [r.y], 'z': [r.z]})),
                                axis=1)

        return df

    def __add_branch_order(self, start=1, level=0):
        self.n.loc[start, "branch_order"] += level
        n_list = self.n[(self.n.parent == start)].index.tolist()
        if len(n_list) == 1:
            self.__add_branch_order(n_list[0], level)
        elif len(n_list) == 0:
            return
        else:
            for y in n_list:
                self.__add_branch_order(y, level + 1)
        return

    def __add_NodeTypeLabel(self):
        self.n["node_type"] = "g"
        branch_nodes = self.BranchNode()
        terminal_nodes = self.TerminalNode()

        remove_nodes = []
        for node in branch_nodes:
            if node == 1:
                continue
            children_nodes = self.n[self.n.parent == node]
            if children_nodes.iloc[0].type != children_nodes.iloc[1].type:
                remove_nodes.append(node)
        for rn in remove_nodes:
            branch_nodes.remove(rn)

        for i in range(1, len(self.n) + 1):
            if i in branch_nodes:
                self.n.loc[i, "node_type"] = "b"
        for j in range(1, len(self.n) + 1):
            if j in terminal_nodes:
                self.n.loc[j, "node_type"] = "t"

        self.n.loc[1, "node_type"] = "s"

    def __NodeTypeCon(self):
        _node_type = ["g", "t", "b", "s"]
        node_type = {}
        for nt in _node_type:
            node_type[nt] = self.n[self.n.node_type == nt]
        return node_type

    def __add_vector(self, df):
        res = self.get_segment(df)
        df['rho'] = res.rho
        df['phi'] = res.phi
        df['theta'] = res.theta
        return df

    def BranchNode(self):  # 返回所有分支节点
        n_d_bns = []
        temp1 = self.n.parent.value_counts()
        self.n_d_bns = []
        for i, j in zip(temp1.index, temp1):
            if j >= 2:
                n_d_bns.append(i)
        n_d_bns.sort()
        return n_d_bns

    def TerminalNode(self):  # 返回所有叶子结点
        child = self.n[self.n.parent != (-1)]
        parent = self.n.loc[child.parent]
        n_d_tns = []
        for x in child.index:
            if x not in parent.index:
                n_d_tns.append(x)
        n_d_tns.sort()
        return n_d_tns

    def get_segment(self, df):  # 返回rho、theta、phi
        def cart2pol_3d(x):
            infinitesimal = 1e-10
            rho = np.sqrt(np.sum(np.square(x), axis=1))
            theta = np.arccos(x[:, 2] / (rho + infinitesimal))
            phi = np.arctan2(x[:, 1], x[:, 0] + infinitesimal)
            return (rho, theta, phi)

        child = df[df.parent.isin(df.index)]
        parent = df.loc[child.parent]
        # child = df[df.parent != (-1)]
        # parent = df.loc[list(child.parent)]
        rho, theta, phi = cart2pol_3d(np.array(child[["x", "y", "z"]]) - np.array(parent[["x", "y", "z"]]))
        res = pd.DataFrame({"type": child.type,
                            "x": (np.array(child.x)),
                            "y": (np.array(child.y)),
                            "z": (np.array(child.z)),
                            "rho": rho,
                            "theta": theta,
                            "phi": phi},
                           index=child.index.tolist())

        # soma
        soma = df[(df.parent == -1)]
        if len(soma) > 0:
            soma_res = pd.DataFrame({"type": 1,
                                     "x": soma.x.iloc[0],
                                     "y": soma.y.iloc[0],
                                     "z": soma.z.iloc[0],
                                     "rho": 1,
                                     "theta": 0,
                                     "phi": 0},
                                    index=[1])
            #     soma.index=s.n.index
            res = soma_res.append(res)
        else:
            print(self.path.split('/')[-1], " soma length: 0")
        return res

    def del_small_segments(self):
        leaves = self.n[self.n.node_type == 't']  # 从叶子结点向上追起，累加长度
        branch_nodes = self.BranchNode()  # 到分支点截止，停止累加长度
        for i in leaves.index.tolist():
            length_path = self.n.loc[i, 'rho']
            path = [i]
            cur = self.n.loc[i]
            #             print('leaves: ',i)
            while (cur.parent not in branch_nodes):
                length_path = length_path + cur.rho
                cur = self.n.loc[cur.parent]
                path.append(cur.parent)
            path.append(cur.parent)
            #

            if length_path <= 10:
                #                 print(length_path,path)
                for i in path:
                    if i != path[-1]:
                        self.n.drop(i, axis=0, inplace=True)
                    else:
                        if len(self.n[self.n.parent == i]) == 0:
                            self.n.loc[i, 'node_type'] = 't'
                        else:
                            self.n.loc[i, 'node_type'] = 'g'
        return

    def get_axon_index(self, key='dict'):  # return axon的所有从子节点到根节点的序列（【id，rho】）
        axon = self.axon
        axon_terminal = axon[axon.node_type == "t"]
        terminal_list = axon_terminal.index.tolist()
        axon_list = axon.index.tolist()
        d = {}
        i = 1
        for ter in terminal_list:
            cur_id = ter
            d["seq" + str(i)] = []
            while (cur_id in axon_list):
                cur_rho = axon.loc[cur_id, 'rho']
                d["seq" + str(i)].append([cur_id, cur_rho])
                cur_id = axon.loc[cur_id, 'parent']
            i = i + 1
        if key == 'list':
            l = []
            for k, v in d.items():
                l = l + v
            return l
        if key == 'dict':
            return d

    def get_dendrites_index(self, key='dict'):  # return dendrites的所有从子节点到根节点的序列（【id，rho】）
        dendrites = self.dendrites
        dendrites_terminal = dendrites[dendrites.node_type == "t"]
        terminal_list = dendrites_terminal.index.tolist()
        dendrites_list = dendrites.index.tolist()
        d = {}
        i = 1
        for ter in terminal_list:
            cur_id = ter
            d["seq" + str(i)] = []
            while (cur_id in dendrites_list):
                #     print(cur)
                cur_rho = dendrites.loc[cur_id, 'rho']
                d["seq" + str(i)].append([cur_id, cur_rho])
                cur_id = dendrites.loc[cur_id, 'parent']
            i = i + 1

        if key == 'list':
            l = []
            for k, v in d.items():
                l = l + v
            return l
        if key == 'dict':
            return d

    def get_all_index(self, key='dict'):  # 区分axon和dendrite后，return所有从子节点到根节点的序列（【id，rho】）
        if key == 'list':
            dendrites = self.get_dendrites_index(key='list')
            axon = self.get_axon_index(key='list')
            all_seqs = dendrites + axon
            return all_seqs
        elif key == 'dict':
            all_seqs = {}
            dendrites = self.get_dendrites_index(key='dict')
            axon = self.get_axon_index(key='dict')
            i = 1
            for seq in dendrites.values():
                all_seqs["seq" + str(i)] = seq
                i = i + 1
            for seq in axon.values():
                all_seqs["seq" + str(i)] = seq
                i = i + 1
            return all_seqs

    def trees_index_from_neuron(self):  # 返回dict，当中每个元素分别属于neuron各个stem主干上的所有index
        trees = {}
        key = 'dict'
        all_seqs = self.get_all_index(key)
        for v in all_seqs.values():
            if v[-1][0] not in trees.keys():
                trees[v[-1][0]] = [v]
            elif v[-1][0] in trees.keys():
                trees[v[-1][0]].append(v)
        return trees

    def neuron2BTree(self, keep):  # 将neuron中各个主干表示为BinaryTree
        # print('keep: ', keep)
        trees = self.trees_index_from_neuron()
        neuron_trees = []
        neuron_trees_dict = {}
        for stem in trees.values():
            bt = BinarySearchTree(keep, stem)  # bt=BinarySearchTree(stem,keep=False) # 默认保留grow type的点
            leave_root = leave(bt.root)
            len_root = path_length(bt.root)
            priority = leave_root * len_root  # 先遍历那一个主干：由主干的叶子节点数乘以总的主干path长度决定。优先遍历小的。
            # print('leave_root ',leave_root,'len_root ',len_root,'priority',priority)
            while priority in neuron_trees_dict.keys():
                priority = priority + 1
            neuron_trees_dict[priority] = bt

        for k, v in sorted(neuron_trees_dict.items(), key=lambda item: item[0]):  # 以dict的key大小作为判断：从小到大遍历
            neuron_trees.append(v)  # 从小到大加入主干
        return neuron_trees

    def neuron2df(self):  # 通过遍历所有的神经元二叉树得到神经元序列的id（list），返回按照id（list）排列的df
        neuron_trees = self.neuron2BTree(keep=True)  # 返回的树为带有grow节点的树
        seqs = []
        length_seqs = {}
        flag = []
        for tree in neuron_trees:
            seq = tree.preorder()  # 以叶子结点的个数为依据前序遍历单个子树
            leave_root = leave(tree.root)
            len_root = path_length(tree.root)
            priority = leave_root * len_root
            length_seqs[priority] = seq
            flag.append(seq[-1])

        for i in sorted(length_seqs.items(), key=lambda item: item[0]):
            seqs = seqs + i[1]
        df = self.n.loc[seqs]

        # return seqs
        return df, flag

    def neuron2seqs(self, with_grow='keep', index_itf=False):  # 返回df, 将neuron2df返回的结果进行处理，选择是否discard grow类型的点
        if index_itf:
            print("with index shuffle")
        else:
            print("without index shuffle")
        # discard grow type nodes?
        df, flag = self.neuron2df()  # 返回的树默认带有grow节点，设置class neuron.neuron2df(self)
        df['flag'] = 0
        df.loc[flag, 'flag'] = 1

        if with_grow == 'discard':
            sum_rho = 0
            sum_phi = 0
            sum_theta = 0
            index_list = df.index.tolist()
            for i in index_list:
                if df.at[i, 'node_type'] == 'g':
                    sum_rho = sum_rho + df.at[i, 'rho']
                    sum_phi = sum_phi + df.at[i, 'phi']
                    sum_theta = sum_theta + df.at[i, 'theta']
                sum_phi = sum_phi % (3.1415926 * 2)
                sum_theta = sum_theta % (3.1415926 * 2)
                if (df.at[i, 'node_type'] == 't') or (df.at[i, 'node_type'] == 'b'):
                    df.at[i, 'rho'] = df.at[i, 'rho'] + sum_rho
                    df.at[i, 'phi'] = df.at[i, 'phi'] + sum_phi
                    df.at[i, 'theta'] = df.at[i, 'theta'] + sum_theta
                    sum_rho = 0
                    sum_phi = 0
                    sum_theta = 0
            df = df[df.node_type != 'g']
            # word_dict = {'b': 1, 't': 2}
            # df['node_type'] = [word_dict[x] for x in df['node_type']]

            if index_itf == True:
                random_index = random.sample(df.index.tolist(), len(df['x']))
                df = df.loc[random_index, :]
            return df

        elif with_grow == 'keep':
            if index_itf == True:
                random_index = random.sample(df.index.tolist(), len(df['x']))
                df = df.loc[random_index, :]
            return df

    def display_neuron(self, save_path=None):  # 将序列(一颗整树，包括多个主干)表示为BinaryTree（多个主干对应的多个二叉树）
        neuron_trees = self.neuron2BTree(keep=False)
        # n_trees = len(neuron_trees)
        WEIGHT, HEIGHT = [], []
        x, y = 0, 0
        X, Y = [], []
        for i in neuron_trees:
            x = (np.sum(WEIGHT)) + 0.5 + get_left_width(i.root) * d_hor  # x, y 是第一个要绘制的节点坐标，由其左子树宽度决定
            X.append(x)
            W, H = get_w_h(i.root)
            W = (W) * d_hor
            H = (H) * d_vec
            WEIGHT.append(W), HEIGHT.append(H)
        y = np.max(HEIGHT) + 1
        # print('X: ', X, 'y: ', y)
        # print('WEIGHT: ', WEIGHT, 'HEIGHT: ', HEIGHT)
        fig, ax = plt.subplots(figsize=(np.sum(WEIGHT) + 4, np.sum(HEIGHT) + 4))
        plt.xlim(0, np.sum(WEIGHT) + 4)
        plt.ylim(0, np.max(HEIGHT) + 4)

        for i, t in enumerate(neuron_trees):
            print_tree_by_inorder(t.root, X[i], y, ax)
        plt.plot([X[0], X[-1]], [y, y], 'k-')
        plt.plot([(X[0] + X[-1]) / 2, (X[0] + X[-1]) / 2], [y, y + d_vec], 'k-')

        if save_path:
            plt.title(save_path.split('/')[-1])
            plt.savefig(save_path, dpi=100)
        else:
            plt.show()
