from typing import Tuple
from collections import *
from copy import *


def reverse_digraph_representation(graph: dict) -> dict:
    reverse_digraph = {}
    for key, value in graph.items():
        reverse_digraph[key] = {}

    for key, value in graph.items():
        for inner_key, inner_value in value.items():
            reverse_digraph[inner_key][key] = inner_value

    return reverse_digraph


print(reverse_digraph_representation({0: {1: 20, 2: 4, 3: 20}, 1: {
    2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}))


def modify_edge_weights(rgraph: dict, root: int) -> None:
    modified_graph = rgraph.copy()
    for key, value in modified_graph.items():
        min_weight = float('inf')
        for inner_value in value.values():
            if inner_value < min_weight:
                min_weight = inner_value
        for inner_key, inner_value in value.items():
            value[inner_key] = inner_value - min_weight


print(modify_edge_weights({0: {}, 1: {0: 20, 4: 4}, 2: {0: 4, 1: 2}, 3: {
    0: 20, 2: 8}, 4: {2: 20, 3: 4}, 5: {1: 16, 3: 8}}, 0))


def compute_rdst_candidate(rgraph: dict, root: int) -> dict:
    rdst_candidate = {}
    for key in rgraph.keys():
        rdst_candidate[key] = {}

    for key in rgraph.keys():
        if key == root:
            continue
        if len(rgraph[key]) > 0:
            m = min(rgraph[key].values())
        else:
            continue
        for neighbor in rgraph[key]:
            if rgraph[key][neighbor] == m:
                rdst_candidate[key][neighbor] = rgraph[key][neighbor]
                break

    return rdst_candidate


print(compute_rdst_candidate({0: {}, 1: {0: 20, 4: 4}, 2: {0: 4, 1: 2}, 3: {
    0: 20, 2: 8}, 4: {2: 20, 3: 4}, 5: {1: 16, 3: 8}}, 0))


def compute_cycle(rdst_candidate: dict) -> tuple:
    for key, value in rdst_candidate.items():
        cycle = [key]
        current_node = key
        while len(rdst_candidate[current_node]) > 0:
            next_node = None
            for neighbor in rdst_candidate[current_node]:
                next_node = neighbor
                break
            if next_node == None:
                break
            current_node = next_node
            if current_node in cycle:
                return tuple(cycle[cycle.index(current_node):])
            cycle.append(current_node)

    return tuple()


print(compute_cycle({1: {4: 4}, 2: {1: 2}, 3: {2: 8}, 4: {3: 4}, 5: {3: 8}}))


def contract_cycle(graph: dict, cycle: tuple) -> Tuple[dict, int]:
    outwards = {}
    inwards = {}

    cstar = max(graph.keys()) + 1
    contracted_graph = {}
    for key in graph.keys():
        if key not in cycle:
            contracted_graph[key] = {}
    contracted_graph[cstar] = {}

    for key in graph.keys():
        for nbr, weight in graph[key].items():
            if key not in cycle and nbr not in cycle:
                contracted_graph[key][nbr] = weight
            elif key in cycle and nbr not in cycle:
                if nbr not in outwards:
                    outwards[nbr] = {}
                outwards[nbr][key] = weight
            elif nbr in cycle and key not in cycle:
                if key not in inwards:
                    inwards[key] = {}
                inwards[key][nbr] = weight

    for key, value in outwards.items():
        if key in contracted_graph:
            contracted_graph[cstar][key] = value[min(value, key=value.get)]

    for key, value in inwards.items():
        if key in contracted_graph:
            contracted_graph[key][cstar] = value[min(value, key=value.get)]

    return (contracted_graph, cstar)


print(contract_cycle({0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {
    3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}, (1, 4, 3, 2)))


def expand_graph(graph: dict, rdst_candidate: dict, cycle: tuple, cstar: int) -> dict:
    rdst_candidate_copy = deepcopy(rdst_candidate)
    important_node = None
    expanded_graph = {}
    for key in graph.keys():
        expanded_graph[key] = {}

    for key, value in rdst_candidate_copy.items():
        for nbr, weight in value.items():
            if key != cstar and nbr != cstar:
                expanded_graph[key][nbr] = rdst_candidate[key][nbr]
            elif key == cstar:
                min_weight = -1
                min_node = -1
                for graphnode in cycle:
                    if nbr in graph[graphnode]:
                        if min_weight == -1 or graph[graphnode][nbr] < min_weight:
                            min_weight = graph[graphnode][nbr]
                            min_node = graphnode
                expanded_graph[min_node][nbr] = min_weight
            elif nbr == cstar:
                nodesintocycle = {}
                nodesintocycle[key] = {}
                for cycle_node in cycle:
                    if cycle_node in graph[key]:
                        nodesintocycle[key][cycle_node] = graph[key][cycle_node]
                important_node = min(
                    nodesintocycle[key], key=nodesintocycle[key].get)
                expanded_graph[key][important_node] = graph[key][important_node]

    for counter in range(len(cycle)):
        if (counter == 0 and cycle[-1] == important_node) or cycle[counter - 1] == important_node:
            pass

        else:
            if counter == 0:
                if cycle[counter] not in expanded_graph:
                    expanded_graph[cycle[counter]] = {}
                expanded_graph[cycle[counter]][cycle[-1]
                                               ] = graph[cycle[counter]].get(cycle[-1])
            else:
                if cycle[counter] not in expanded_graph:
                    expanded_graph[cycle[counter]] = {}
                expanded_graph[cycle[counter]][cycle[counter - 1]
                                               ] = graph[cycle[counter]].get(cycle[counter - 1])

    return expanded_graph


print(expand_graph(
    {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20},
     3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}},  # graph
    {0: {6: 4}, 6: {5: 8}},  # rdst_candidate
    (1, 4, 3, 2),  # cycle
    6))  # cstar


def bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.

        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from

        Returns:
        The distances from startnode to each node
    """
    dist = {}

    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize search queue
    queue = deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def compute_rdmst(graph, root):
    """
        This function checks if:
        (1) root is a node in digraph graph, and
        (2) every node, other than root, is reachable from root
        If both conditions are satisfied, it calls compute_rdmst_helper
        on (graph, root).

        Since compute_rdmst_helper modifies the edge weights as it computes,
        this function reassigns the original weights to the RDMST.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node id.

        Returns:
        An RDMST of graph rooted at r and its weight, if one exists;
        otherwise, nothing.
    """

    if root not in graph:
        print("The root node does not exist")
        return

    distances = bfs(graph, root)
    for node in graph:
        if distances[node] == float('inf'):
            print("The root does not reach every other node in the graph")
            return

    rdmst = compute_rdmst_helper(graph, root)

    # reassign the original edge weights to the RDMST and computes the total
    # weight of the RDMST
    rdmst_weight = 0
    for node in rdmst:
        for nbr in rdmst[node]:
            rdmst[node][nbr] = graph[node][nbr]
            rdmst_weight += rdmst[node][nbr]

    return (rdmst, rdmst_weight)


def compute_rdmst_helper(graph, root):
    """
        Computes the RDMST of a weighted digraph rooted at node root.
        It is assumed that:
        (1) root is a node in graph, and
        (2) every other node in graph is reachable from root.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node in graph.

        Returns:
        An RDMST of graph rooted at root. The weights of the RDMST
        do not have to be the original weights.
        """

    # reverse the representation of graph
    rgraph = reverse_digraph_representation(graph)

    # Step 1 of the algorithm
    modify_edge_weights(rgraph, root)

    # Step 2 of the algorithm
    rdst_candidate = compute_rdst_candidate(rgraph, root)

    # compute a cycle in rdst_candidate
    cycle = compute_cycle(rdst_candidate)

    # Step 3 of the algorithm
    if not cycle:
        return reverse_digraph_representation(rdst_candidate)
    else:
        # Step 4 of the algorithm

        g_copy = deepcopy(rgraph)
        g_copy = reverse_digraph_representation(g_copy)

        # Step 4(a) of the algorithm
        (contracted_g, cstar) = contract_cycle(g_copy, cycle)
        # cstar = max(contracted_g.keys())

        # Step 4(b) of the algorithm
        new_rdst_candidate = compute_rdmst_helper(contracted_g, root)

        # Step 4(c) of the algorithm
        rdmst = expand_graph(reverse_digraph_representation(
            rgraph), new_rdst_candidate, cycle, cstar)

        return rdmst


def find_first_positives(trace_data):
    """
        Finds the first positive test date of each patient
        in the trace data.
        Arguments:
        trace_data -- a list of data pertaining to location
        and first positive test date
        Returns:
        A dictionary with patient id's as keys and first positive
        test date as values. The date numbering starts from 0 and
        the patient numbering starts from 1.
        """
    first_pos = {}
    for pat in range(len(trace_data[0])):
        first_pos[pat + 1] = None
        for date in range(len(trace_data)):
            if trace_data[date][pat].endswith(".5"):
                first_pos[pat + 1] = date
                break
    return first_pos


def compute_genetic_distance(sequence1, sequence2) -> int:
    sum = 0
    for counter1, counter2 in zip(sequence1, sequence2):
        if counter1 != counter2:
            sum += 1
    return sum


def read_patient_sequences(filename):
    """
        Turns the bacterial DNA sequences (obtained from patients) into a list containing tuples of
        (patient ID, sequence).

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A list of (patient ID, sequence) tuples.
        """
    sequences = []
    with open(filename) as f:
        line_num = 0
        for line in f:
            if len(line) > 5:
                patient_num, sequence = line.split("\t")
                sequences.append((int(patient_num), ''.join(
                    e for e in sequence if e.isalnum())))
    return sequences


def read_patient_traces(filename):
    """
        Reads the epidemiological data file and computes the pairwise epidemiological distances between patients

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A dictionary of dictionaries where dict[i][j] is the
        epidemiological distance between i and j.
    """
    trace_data = []
    patient_ids = []
    first_line = True
    with open(filename) as f:
        for line in f:
            if first_line:
                patient_ids = line.split()
                patient_ids = list(map(int, patient_ids))
                first_line = False
            elif len(line) > 5:
                trace_data.append(line.rstrip('\n'))
    return compute_pairwise_epi_distances(trace_data, patient_ids)


def compute_pairwise_gen_distances(sequences, distance_function):
    """
        Computes the pairwise genetic distances between patients (patients' isolate genomes)

        Arguments:
        sequences -- a list of sequences that correspond with patient id's
        distance_function -- the distance function to apply to compute the weight of the 
        edges in the returned graph

        Returns:
        A dictionary of dictionaries where gdist[i][j] is the
        genetic distance between i and j.
        """
    gdist = {}
    cultures = {}

    # Count the number of differences of each sequence
    for i in range(len(sequences)):
        patient_id = sequences[i][0]
        seq = sequences[i][1]
        if patient_id in cultures:
            cultures[patient_id].append(seq)
        else:
            cultures[patient_id] = [seq]
            gdist[patient_id] = {}
    # Add the minimum sequence score to the graph
    for pat1 in range(1, max(cultures.keys()) + 1):
        for pat2 in range(pat1 + 1, max(cultures.keys()) + 1):
            min_score = float("inf")
            for seq1 in cultures[pat1]:
                for seq2 in cultures[pat2]:
                    score = distance_function(seq1, seq2)
                    if score < min_score:
                        min_score = score
            gdist[pat1][pat2] = min_score
            gdist[pat2][pat1] = min_score
    return gdist


def compute_epi_distance(pid1, pid2, trace_data, first_pos1, first_pos2, patient_ids):
    """
        Computes the epidemiological distance between two patients.

        Arguments:
        pid1 -- the assumed donor's index in trace data
        pid2 -- the assumed recipient's index in trace data
        trace_data -- data for days of overlap and first positive cultures
        first_pos1 -- the first positive test day for pid1
        first_pos2 -- the first positive test day for pid2
        patient_ids -- an ordered list of the patient IDs given in the text file

        Returns:
        Finds the epidemiological distance from patient 1 to
        patient 2.
        """
    first_overlap = -1
    assumed_trans_date = -1
    pid1 = patient_ids.index(pid1)
    pid2 = patient_ids.index(pid2)
    # Find the first overlap of the two patients
    for day in range(len(trace_data)):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            first_overlap = day
            break
    if (first_pos2 < first_overlap) | (first_overlap < 0):
        return len(trace_data) * 2 + 1
    # Find the assumed transmission date from patient 1 to patient 2
    for day in range(first_pos2, -1, -1):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            assumed_trans_date = day
            break
    sc_recip = first_pos2 - assumed_trans_date

    if first_pos1 < assumed_trans_date:
        sc_donor = 0
    else:
        sc_donor = first_pos1 - assumed_trans_date
    return sc_donor + sc_recip


def compute_pairwise_epi_distances(trace_data, patient_ids):
    """
        Turns the patient trace data into a dictionary of pairwise 
        epidemiological distances.

        Arguments:
        trace_data -- a list of strings with patient trace data
        patient_ids -- ordered list of patient IDs to expect

        Returns:
        A dictionary of dictionaries where edist[i][j] is the
        epidemiological distance between i and j.
        """
    edist = {}
    proc_data = []
    # Reformat the trace data
    for i in range(len(trace_data)):
        temp = trace_data[i].split()[::-1]
        proc_data.append(temp)
    # Find first positive test days and remove the indication from the data
    first_pos = find_first_positives(proc_data)
    for pid in first_pos:
        day = first_pos[pid]
        proc_data[day][pid - 1] = proc_data[day][pid - 1].replace(".5", "")
    # Find the epidemiological distance between the two patients and add it
    # to the graph
    for pid1 in patient_ids:
        edist[pid1] = {}
        for pid2 in patient_ids:
            if pid1 != pid2:
                epi_dist = compute_epi_distance(pid1, pid2, proc_data,
                                                first_pos[pid1], first_pos[pid2], patient_ids)
                edist[pid1][pid2] = epi_dist
    return edist


def construct_complete_weighted_digraph(gen_file: str, epi_file: str):
    patient_sequences = read_patient_sequences(gen_file)
    patient_traces = read_patient_traces(epi_file)
    return_graph = {}

    for patient in patient_sequences:
        return_graph[patient[0]] = {}

    maximum = float('-inf')
    for patient in patient_traces:
        for diff_patient in patient_traces[patient]:
            if patient_traces[patient][diff_patient] > maximum:
                maximum = patient_traces[patient][diff_patient]

    genetic_distances = compute_pairwise_gen_distances(
        patient_sequences, compute_genetic_distance)

    for patient in return_graph:
        for diff_patient in return_graph:
            if patient != diff_patient:
                g_ab = genetic_distances[patient][diff_patient]
                e_ab = patient_traces[patient][diff_patient]
                weight = g_ab + ((999 * (e_ab / maximum)) / 100000)
                return_graph[patient][diff_patient] = weight

    return return_graph


def infer_transmap(gen_data, epi_data, patient_id):
    complete_digraph = construct_complete_weighted_digraph(gen_data, epi_data)
    return compute_rdmst(complete_digraph, patient_id)


print(infer_transmap("patient_sequences.txt", "patient_traces.txt", 1))
