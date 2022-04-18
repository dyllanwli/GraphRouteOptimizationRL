import numpy as np


def get_policy_iteration(env, gamma=0.99999999999):
    R = env.R
    P = env.P
    nstates, nactions = P.shape[-1], P.shape[0]

    def Pd(d):
        pd = np.einsum('ast,sa->st', P, d)
        return pd

    def rd(d):
        return np.einsum('sa,sa->s', R, d)

    def bellman_content(v):
        x = np.einsum("ast,t->sa", P, v)
        return R + gamma * x

    def policy_evaluation_pio(d):
        gamma_pd = gamma * Pd(d)
        gamma_pd = np.identity(nstates) - gamma_pd
        return np.linalg.solve(gamma_pd, rd(d))

    def bellman_policy(v):  # looks weird because we want the policy as an array! not a vector
        d_next = np.zeros((nstates, nactions))
        cont = bellman_content(v)
        best_actions = np.argmax(cont, axis=1)
        d_next[list(range(len(best_actions))), best_actions] = 1
        return d_next

    def policy_iteration_operator(d):
        v = policy_evaluation_pio(d)
        return v, bellman_policy(v)

    guess = np.identity(len(env.graph.ngraph.nodes))

    old_guess = None
    while True:
        if np.all(guess == old_guess):
            break

        old_guess = guess
        guess = policy_iteration_operator(guess)

        guess = guess[1].astype(int)

    return guess

def get_backward_induction_actions(fin_env):
    """
    The env must be one created with make_horizon=True
    """
    nb_nodes = fin_env.graph.ngraph.order()

    def bellman_content(v):
        x = np.einsum("ast,t->sa", fin_env.P, v)
        return fin_env.R + x

    def bellman_optimality_operator(v):
        return np.max(bellman_content(v), axis=1)

    def bellman_policy_operator(v):
        return np.argmax(bellman_content(v), axis=1)

    vts = [np.zeros((nb_nodes,))]  # start with only the final v, a vector of zeros
    pts = []

    for t in range(len(fin_env.horizon_states)):
        v_tm1 = vts[-1]
        v_t = bellman_optimality_operator(v_tm1)
        p_t = bellman_policy_operator(v_tm1)
        pts.append(p_t)
        vts.append(v_t)

    sequence = []
    position = fin_env.graph.origin
    for epoch in reversed(pts):
        sequence.append(epoch[position])
        position = epoch[position]
    return sequence
