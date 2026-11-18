"""
This module handles the serving of a QUBO instance.
It supports receiving a QUBO instance, and returns it as another data-structure.

Input:
(currently)
- ConvertedQubo: which is obtained from:
    1. raw_qubo + parsing = QuboInstance
    2. QuboInstance  + source conventions  = InterpretedQubo
    3. InterpretedQubo + target conventions =  ConvertedQubo


Output:
    (currently)
    1. ConvertedQubo ---> BQM :: to_dimod_bqm()
    2. ConvertedQubo ---> NX.GRAPH :: to_networkx_graph()
    3. ConvertedQubo ---> TEMP_FILE :: to_tmp_file()
    4. ConvertedQubo ---> OBJECTIVE_FUNC :: to_objective_function()
    5. ConvertedQubo ---> FUTURE: new data structures should register their adapter to this module.


"""
import dimod
from .conventions import ConvertedQubo

####-----------------------------AS-BQM-----------------------------------------####

def to_dimod_bqm(converted):
    """
    Convert a ready-to-use ConvertedQubo into a dimod BQM.

    Input:
        ConvertedQubo in upper_triangular, minimize convention.

    Output:
        dimod.BinaryQuadraticModel with BINARY variables.

    Role:
        Adapt the data structure only. No mathematical convention
        conversion or source interpretation is performed here.

    Notes:
        - Keep the original 1-based variable labels.
        - Include all declared variables, even isolated ones.
        - Convert Decimal coefficients to float for dimod.
    """
    if not isinstance(converted, ConvertedQubo):
        raise TypeError("Expected a ConvertedQubo")

    if converted.convention != "upper_triangular":
        raise ValueError("BQM adapter requires upper_triangular convention")

    if converted.direction != "minimize":
        raise ValueError("BQM adapter requires minimize direction")

    # Initialize every declared variable, even if it has no bias.
    linear = {
        i: 0.0
        for i in range(1, converted.num_variables + 1)
    }
    quadratic = {}

    # Separate diagonal terms from pairwise interactions.
    for (i, j), coefficient in converted.coefficients.items():
        if i == j:
            linear[i] = float(coefficient)
        else:
            quadratic[(i, j)] = float(coefficient)

    # Construct the actual solver-compatible data structure.
    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        float(converted.offset),
        dimod.BINARY,
    )

    return bqm

####----------------------------AS-NX-GRAPH--------------------------------------####


def to_networkx_graph(converted, *, include_metadata=False):
    """
    Convert a 'ConvertedQubo' PBJECT into a NetworkX interaction graph.

    Input:
        A mathematically converted QUBO instance.

    Output:
        An undirected nx.Graph with all declared variables
        and all nonzero pairwise interactions.

    If include_metadata=True:
        - Node attributes: linear coefficients.
        - Edge attributes: quadratic coefficients.
        - Graph attributes: source identity, convention,
          direction, offset and other essential metadata.

    Role:
        Change the data structure only.
        No coefficient scaling or direction conversion occurs.

    Notes:
        Variable labels remain 1-based.
        Missing linear coefficients are implicitly zero.
        Stored edge coefficients follow converted.convention.
    """
    if not isinstance(converted, ConvertedQubo):
        raise TypeError("Expected a ConvertedQubo")

    import networkx as nx

    graph = nx.Graph()

    # Preserve every declared variable, including isolated ones.
    graph.add_nodes_from(range(1, converted.num_variables + 1))

    # Build the graph from the converted QUBO coefficients.
    for (i, j), coefficient in converted.coefficients.items():

        if i == j:
            # Diagonal QUBO terms are node properties, not edges.
            if include_metadata:
                graph.nodes[i]["linear_coefficient"] = coefficient

        elif coefficient != 0:
            # Off-diagonal terms define the interaction graph.
            graph.add_edge(i, j)

            if include_metadata:
                graph.edges[i, j]["quadratic_coefficient"] = coefficient

    # Attach metadata only when explicitly requested.
    if include_metadata:
        raw = converted.interpreted.raw

        graph.graph.update(
            instance_id=raw.instance_id,
            family=raw.family,
            source_path=str(raw.source_path),
            num_variables=converted.num_variables,
            index_base=1,
            convention=converted.convention,
            direction=converted.direction,
            offset=converted.offset,
            energy_sign=converted.energy_sign,
        )

    return graph


####----------------------------AS-OBJECTIVE-FNC---------------------------------####

def to_objective_function(converted):
    """
    Return a callable objective function from a ConvertedQubo.

    Input:
        ConvertedQubo with its target convention and direction
        already defined.

    Output:
        A callable f(assignment) -> Decimal energy.

    Role:
        Evaluate the converted QUBO objective, including its offset.
        No convention conversion or direction inversion is performed.

    Assignment:
        - A sequence of n binary values, in variable order 1..n.
        - Or a dictionary mapping variable labels 1..n to binary values.
    """
    if not isinstance(converted, ConvertedQubo):
        raise TypeError("Expected a ConvertedQubo")

    n = converted.num_variables

    def objective(assignment):
        # Convert the assignment into a dictionary indexed from 1.
        if isinstance(assignment, dict):
            if set(assignment) != set(range(1, n + 1)):
                raise ValueError(f"Assignment must have exactly keys 1..{n}")
            bits = assignment
        else:
            if len(assignment) != n:
                raise ValueError(f"Assignment must have {n} bits")
            bits = {
                i: bit
                for i, bit in enumerate(assignment, start=1)
            }

        # Every variable must receive a valid binary value.
        if any(type(bit) is not int or bit not in (0, 1)
               for bit in bits.values()):
            raise ValueError("Assignment values must be integer 0 or 1")

        # Evaluate the objective, starting with its constant offset.
        energy = converted.offset

        for (i, j), coefficient in converted.coefficients.items():
            # In a symmetric matrix, each off-diagonal coefficient
            # contributes twice to x.T @ Q @ x.
            multiplier = (
                2 if i != j and converted.convention == "symmetric"
                else 1
            )

            energy += coefficient * multiplier * bits[i] * bits[j]

        return energy

    return objective


####----------------------------AS-TMP-FILE--------------------------------------####

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .conventions import ConvertedQubo


@contextmanager
def to_mqlib_file(converted):
    """
    Create a temporary QUBO input file for MQLib.

    Input:
        ConvertedQubo with:
            convention = "symmetric"
            direction = "maximize"

    Output:
        Path to a temporary MQLib-compatible QUBO file.

    Role:
        Serialize the already-converted coefficients.
        No mathematical transformations are performed.

    Notes:
        - Variable indices are 1-based.
        - Each matrix entry is written once (i <= j).
        - The offset is not stored in the MQLib file.
        - The file exists only inside the 'with' block.
    """
    if not isinstance(converted, ConvertedQubo):
        raise TypeError("Expected a ConvertedQubo")

    if converted.convention != "symmetric":
        raise ValueError("MQLib requires symmetric convention")

    if converted.direction != "maximize":
        raise ValueError("MQLib requires maximize direction")

    # Collect the nonzero coefficients in a predictable order.
    entries = sorted(
        (i, j, coefficient)
        for (i, j), coefficient in converted.coefficients.items()
        if coefficient != 0
    )

    # Create a unique temporary directory.
    # It will be removed automatically when the block ends.
    with TemporaryDirectory(prefix="mqlib_qubo_") as directory:
        path = Path(directory) / "instance.qubo"

        with path.open("w", encoding="utf-8") as output:
            # First line: number of variables, number of entries.
            output.write(
                f"{converted.num_variables} {len(entries)}\n"
            )

            # Remaining lines: i, j, matrix coefficient.
            for i, j, coefficient in entries:
                output.write(f"{i} {j} {coefficient}\n")

        # The caller may use the file until it leaves the block.
        yield path
