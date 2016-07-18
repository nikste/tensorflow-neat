from Constants import OUTPUT0, OUTPUT1, INPUT1, INPUT0
from data_fetcher import get_gaussian_quantiles
from tensorflow_utils import build_and_test
import sys
import numpy as np


def add_connection(connections, genotype):
    enabled_innovations = [k for k in genotype.keys() if genotype[k]]

    enabled_connections = [connections[cns] for cns in enabled_innovations]

    # get reachable nodes
    froms = set([fr[1] for fr in enabled_connections ])
    tos = set([to[2] for to in enabled_connections])

    nodes = sorted(list(froms.union(tos)))

    # select random two:
    r1 = np.random.randint(0,len(nodes))
    r2 = np.random.randint(0,len(nodes) - 1)
    if r2 >= r1:
        r2 += 1

    r1 = nodes[r1]
    r2 = nodes[r2]
    from_node = r2 if r2 < r1 else r1
    to_node = r2 if r2 > r1 else r1

    assert(from_node < to_node)

    # prevent connections from input to input nodes and output to output nodes.
    # todo change this
    if from_node == INPUT0 and to_node == INPUT1 or from_node == OUTPUT0 and to_node == OUTPUT1:
        return add_connection( connections, genotype)

    # check if connection already there
    if not any(from_node == c[1] and to_node == c[2] for c in connections):
        connections.append((len(connections), from_node, to_node))

        genotype[len(connections) - 1 ] = True

    assert(len(genotype.keys()) <= len(connections))
    return connections, genotype


def add_node(connections, genotype, debug=False):
    # select random connection that is enabled
    enabled_innovations = [k for k in genotype.keys() if genotype[k]]

    # get random connection:
    r = np.random.randint(0,len(enabled_innovations))
    connections_innovation_index = enabled_innovations[r]
    connection_to_split = connections[connections_innovation_index]

    from_node = connection_to_split[1]
    to_node = connection_to_split[2]

    new_node = (to_node - from_node) / 2 + from_node

    if debug:
        print "from:", from_node
        print "to:", to_node
        print "new:", new_node
    # todo: what to do if node id already exist? -> just leave it be.

    # add two new connection items: from_node -> new_node; new_node -> to_node
    # check if already existing beforehand.
    # todo: there should be a smarter way to do this than just give up.
    if not from_node < new_node:
        return connections, genotype
    if not new_node < to_node:
        return connections, genotype
    assert(from_node < new_node)
    assert(new_node < to_node)
    # check from to
    if not any(from_node == c[1] and new_node == c[2] for c in connections):
        id = len(connections)
        connections.append((id, from_node, new_node))
        genotype[id] = True
    else:
        ind = [c[0] for c in connections if c[1] == from_node and c[2] == new_node]
        genotype[ind[0]] = True

    if not any(new_node == c[1] and to_node == c[2] for c in connections):
        id = len(connections)
        connections.append((id, new_node, to_node))
        genotype[id] = True
    else:
        ind = [c[0] for c in connections if new_node == c[1] and to_node == c[2]]
        genotype[ind[0]] = True

    # add new node

    # disable old connection where we now inserted a new node
    genotype[connections_innovation_index] = False

    assert (len(genotype.keys()) <= len(connections))

    return connections, genotype




def crossover(connections, genotype0, performance0 , genotype1, performance1):
    # 1. matching genes are inherited at random (everything is made up and the weights don't matter here)
    # 2. disjoint and excess from the more fit parent
    # 3. preset chance to disable gene if its disabled in either parent

    # new genes should be always in the end
    k_0 = sorted(genotype0.keys())
    k_1 = sorted(genotype1.keys())

    # inherit disjoint from more fit parent
    offspring_genotype = {}
    if performance0 > performance1 and len(k_0) > len(k_1):
        # 0 is better and has more genes
        for l in connections:
            innovation_num = l[0]
            if innovation_num in k_0:
                offspring_genotype[innovation_num] = genotype0[innovation_num]
            elif innovation_num in k_1:
                offspring_genotype[innovation_num] = genotype1[innovation_num]

                # if one of them is disabled, disable (maybe in child)
        # for k in k_1:
        #     if genotype0[k] == False or genotype1[k] == False:
        #         r = np.random.randint(0,2)
        #         if r == 1:
        #                 offspring_genotype[k] = False
    elif performance1 > performance0 and len(k_1) > len(k_0):
        for l in connections:
            innovation_num = l[0]
            if innovation_num in k_1:
                offspring_genotype[innovation_num] = genotype1[innovation_num]
            elif innovation_num in k_0:
                offspring_genotype[innovation_num] = genotype0[innovation_num]
        # for k in k_0:
            # if genotype0[k] == False or genotype1[k] == False:
            #     r = np.random.randint(0, 2)
            #     if r == 1:
            #         offspring_genotype[k] = False

    elif len(k_1) < len(k_0):
        for k in k_1:
            offspring_genotype[k] = genotype1[k]
            # if genotype0[k] == False or genotype1[k] == False:
            #     r = np.random.randint(0, 2)
            #     if r == 1:
            #         offspring_genotype[k] = False
    elif len(k_0) <= len(k_1):
        for k in k_0:
            offspring_genotype[k] = genotype0[k]
            # if genotype0[k] == False or genotype1[k] == False:
            #     r = np.random.randint(0, 2)
            #     if r == 1:
            #         offspring_genotype[k] = False

    return offspring_genotype
    # add sorted keys anyway
    # check disjoint and add if the one that has them is stronger
    #
    # performance0, performance1


def eval_fitness(connections, genotype, x, y, x_test, y_test, run_id="1"):
    # todo: only tests train
    perf_train = build_and_test(connections, genotype, x, y, x_test, y_test, run_id=run_id)
    # print perf_test,perf_train
    return perf_train

def start_neuroevolution(x, y, x_test, y_test):
    """starts neuroevolution on binary dataset"""

    connections = [(0, INPUT0, OUTPUT0), (1, INPUT1, OUTPUT0), (2, INPUT0, OUTPUT1), (3, INPUT1, OUTPUT1)]
    genotypes = [{0: True, 1: True, 2: True, 3: True} for d in xrange(5)]

    for its in xrange(0,100):
        print "iteration", its

        fitnesses = []
        # test networks
        for i in xrange(0,len(genotypes)):
            fitnesses.append(eval_fitness(connections, genotypes[i], x, y, x_test, y_test, run_id=str(its) + "/" + str(i)))

        # get indices of sorted list
        fitnesses_sorted_indices = [i[0] for i in reversed(sorted(enumerate(fitnesses), key=lambda x: x[1]))]

        print "connections:\n"
        print connections
        for ra in xrange(0,len(fitnesses_sorted_indices)):
            print fitnesses[fitnesses_sorted_indices[ra]], genotypes[fitnesses_sorted_indices[ra]]

        # run evolutions
        # todo: fiddle with parameters, include size of network in fitness?
        new_gen = []
        # copy five best survivors already
        m = 5
        if m > len(fitnesses):
            m = len(fitnesses)

        for i in xrange(0,m):
            print "adding:", fitnesses[fitnesses_sorted_indices[i]], genotypes[fitnesses_sorted_indices[i]]
            new_gen.append(genotypes[fitnesses_sorted_indices[i]])

        for i in xrange(0,len(fitnesses_sorted_indices)):
            fi = fitnesses_sorted_indices[i]
            r = np.random.uniform()
            # select the best for mutation and breeding, kill of worst.
            if r <= 0.2:
                # mutate
                connections, gen = add_connection(connections, genotypes[i])
                new_gen.append(gen)
            r = np.random.uniform()
            if r <= 0.5:
                connections, gen = add_node(connections, genotypes[i])
                new_gen.append(gen)

            r = np.random.uniform()
            if r <= 0.1:
                # select random for breeding
                r = np.random.randint(0,len(fitnesses))
                r2 = np.random.randint(0,len(fitnesses) - 1)
                if r2 >= r:
                    r2 +=1
                gen = crossover(connections, genotypes[r], fitnesses[r], genotypes[r2], fitnesses[r2])
                new_gen.append(gen)
                new_gen.append(genotypes[fi])
                # stop if we have 5 candidates
            # new_gen.append(genotypes[fi])
            if len(new_gen) > 10:
                genotypes = new_gen
                break

        # kill off all but 4 best
        # # todo: change
        # c = 0
        # for i in range(0,len(fitnesses_sorted_indices)):
        #     new_gen.append(genotypes[fitnesses_sorted_indices[i]])

        genotypes = new_gen




def test_crossover2():
    c = [(0, 0, 10000), (1, 1, 10000), (2, 0, 10001), (3, 1, 10001), (4, 0, 5000), (5, 5000, 10001), (6, 5000, 10000), (7, 5000, 7500), (8, 7500, 10001), (9, 1, 5000)]
    g0 = {0: True, 1: False, 2: True, 3: True, 6: True, 9: True}
    p0 = 0.381314
    g1 = {0: True, 1: True, 2: True, 3: True}
    p1 = 0.371789

    offspring = crossover(c,g0,p0,g1,p1)
    ## {0: True, 1: False, 2: True, 3: True, 6: True, 9: True}

def test_crossover():
    x, y = get_gaussian_quantiles(n_samples=100)
    x_test, y_test = get_gaussian_quantiles(n_samples=100)


    connections = [(0, INPUT0, OUTPUT0), (1, INPUT1, OUTPUT0), (2, INPUT0, OUTPUT1), (3, INPUT1, OUTPUT1)]

    # genotype0 = {0: True, 1: True, 2: True, 3: True}
    # genotype1 = {0: True, 1: True, 2: True, 3: True}
    genotype0 = {0: True, 1: True, 2: True, 3: True}
    genotype1 = {0: True, 1: True, 2: True, 3: True}
    connections, genotype0 = add_node(connections, genotype0)

    perf0 = 0.6
    perf1 = 0.5
    print connections
    print genotype0
    print perf0
    print genotype1
    print perf1
    offspringgenotype = crossover(genotype0, perf0, genotype1, perf1)
    print offspringgenotype





if __name__ == "__main__":
    #test_crossover2()
    # test_crossover()

    x, y = get_gaussian_quantiles(n_samples=1000)
    x_test, y_test = get_gaussian_quantiles(n_samples=100)


    connections = [(0, INPUT0, OUTPUT0), (1, INPUT1, OUTPUT0), (2, INPUT0, OUTPUT1), (3, INPUT1, OUTPUT1)]

    genotype0 = {0:True, 1:True, 2:True, 3:True}
    genotype1 = {0:True, 1:True, 2:True, 3:True}
    # genotype1 = {}
    for i in xrange(0,5000):

        print "adding_node: gen0:",genotype0
        connections, genotype0 = add_node(connections, genotype0)
        print "after: gen0",genotype0


        print "genotype0"
        perf1 = eval_fitness(connections, genotype0, x, y, x_test, y_test, iteration=i)

        i += 1
        print "genotype1"
        perf0 = eval_fitness(connections, genotype1, x, y, x_test, y_test, iteration=i)
        print "adding_connection: gen1",genotype1
        connections, genotype1 = add_connection(connections, genotype1)
        print "after: gen1",genotype1

        print "connections:",connections
        print "crossover:"
        print "gen0, perf0", genotype0, perf0
        print "gen1, perf1", genotype1, perf1
        genotype0 = crossover(connections, genotype0,perf0,genotype1,perf1)
        print "after:",genotype0

    #
    # perf = eval_fitness(connections, genotype, x, y, x_test, y_test)
    # print "performance:", perf



