

    # def _transform_to_horizon(self, ngraph, origin, goal):
    #     all_paths = sorted(nx.all_simple_paths(
    #         ngraph, origin, goal), key=len, reverse=True)
    #     max_len = len(all_paths[0])
    #     adj_mat = get_correct_adj_mat(ngraph)

    #     _new_name = [ngraph.order() + 1]

    #     def get_new_name(_new_name=_new_name, check=True):
    #         temp = _new_name[0]
    #         if check:
    #             while temp == goal or temp == origin:
    #                 temp += 1
    #         _new_name[0] = temp + 1
    #         return temp

    #     # go-go gadgeto extendo paths
    #     for path in all_paths:
    #         while len(path) < max_len:
    #             path.insert(-1, get_new_name())

    #     # rename paths
    #     flow_graph = nx.DiGraph()
    #     flow_graph.add_node(origin)
    #     flow_graph.add_node(goal)
    #     for path in all_paths:
    #         new_u_name = origin
    #         for i, uv in enumerate(make_bigram(path)):
    #             u, v = uv
    #             new_v_name = v
    #             if v != goal:
    #                 new_v_name = get_new_name()
    #                 flow_graph.add_node(new_v_name)

    #             w = 0
    #             if u < ngraph.order():
    #                 if v < ngraph.order():
    #                     w = adj_mat[u, v]
    #                 else:
    #                     w = adj_mat[u, goal]
    #             flow_graph.add_edge(new_u_name, new_v_name, weight=w)
    #             new_u_name = new_v_name

    #     # collapse end
    #     front = goal
    #     while True:
    #         neighs = list(flow_graph.predecessors(front))
    #         for neigh in neighs.copy():
    #             if flow_graph[neigh][front]["weight"] != 0:
    #                 neighs.remove(neigh)

    #         if len(neighs) <= 1:
    #             break

    #         front = neighs[0]
    #         for neigh in neighs[1:]:
    #             for pred in flow_graph.predecessors(neigh):
    #                 flow_graph.add_edge(
    #                     pred, front, weight=flow_graph[pred][neigh]["weight"])
    #             flow_graph.remove_node(neigh)

    #     # final_relabeling
    #     dont_rename_poi = True
    #     if origin > flow_graph.order()-1 or goal > flow_graph.order()-1:
    #         dont_rename_poi = False

    #     rename_origin = origin
    #     rename_goal = goal

    #     _new_name[0] = 0
    #     for n in list(flow_graph.nodes):
    #         if dont_rename_poi and n in [origin, goal]:
    #             continue

    #         new_name = get_new_name(check=dont_rename_poi)
    #         if not dont_rename_poi:
    #             if n == origin:
    #                 rename_origin = new_name
    #             if n == goal:
    #                 rename_goal = new_name

    #         nx.relabel_nodes(flow_graph, {n: new_name}, copy=False)

    #     return flow_graph, rename_origin, rename_goal

    # def _make_horizons(self, ngraph, origin, goal):
    #     def test_if_already_ok():
    #         visited = {origin}
    #         front = [origin]
    #         while True:
    #             if len(front) == 0:
    #                 break

    #             sucs = []
    #             for f in front:
    #                 sucs += list(ngraph.successors(f))
    #             if len(sucs) == 0:
    #                 break

    #             suc_set = set(sucs)

    #             if len(suc_set.intersection(visited)) > 0:
    #                 return False

    #             front = list(suc_set)
    #             visited = visited.union(suc_set)

    #         # true
    #         if ngraph.order() > len(visited):
    #             not_visited = set(ngraph.nodes) - visited
    #             for nv in not_visited:
    #                 ngraph.remove_node(nv)

    #         return True

    #     if not test_if_already_ok():
    #         ngraph, origin, goal = self._transform_to_horizon(
    #             ngraph, origin, goal)

    #     self.horizon_acts = []
    #     self.horizon_states = []

    #     front = [origin]
    #     while True:
    #         self.horizon_states.append(front)
    #         all_sucs = set()
    #         acts_per_state = []
    #         self.horizon_acts.append(acts_per_state)
    #         for node in front:
    #             sucs = list(ngraph.successors(node))
    #             acts_per_state.append(sucs)
    #             all_sucs = all_sucs.union(set(sucs))
    #         if len(all_sucs) == 0:
    #             break
    #         front = list(all_sucs)

    #     return ngraph, origin, goal