"""
QUBO conventions: interpret the source, then convert to another convention.

This is ONLY the mathematical-conventions layer.
It does NOT create 'Solver-ready-input' objects -
Those belong to a later output/solver-adapters layer.

The end-game is this:
We read QUBO instance under a predefined convension.
Then a solver lands its requirement: opt_direction, matrix_convension, input_type.
Then, out reader navigates these requirements.. meaning that:
reader returns, for example, the insntace:
    - Converted from max to min,
    - Converted from upper-triangulare to symmetric convension
    - Cast the interpreted QUBO into Graph/temp_file/BQM/Objective_func/etc..

So: We are slowly, layer after layer, build a Semantic/conventions/Forms NAVIGATOR.

The raw input:
A .qs file gives us stored triples (i, j, coefficient). By themselves,
those triples do not tell us whether off-diagonal coefficients should be
counted once or twice. We therefore require the caller to choose:

    source_convention: 'symmetric' or 'upper_triangular'
    source_direction:  'minimize' or 'maximize'

We make NO attempt to infer those choices from comments in the file for the time being.
All variable labels remain 1-based. All arithmetic uses Decimal.

"""

from decimal import Decimal

from .model import QuboInstance


class InterpretedQubo:
    """
    TL;DR : this class holds a qubo + its interpretation

    A QUBO understood under one *explicit* interpretation of its raw data.

    Data here represents the function
        E(x) = offset + sum(linear[i] * x[i])
                      + sum(quadratic[i,j] * x[i] * x[j]).

    The quadratic values are actual coefficients of x[i]*x[j].
    They are NOT necessarily the numbers stored in the original .qs file.



    """

    def __init__(self, raw, source_convention, direction,
                 linear, quadratic, offset, assumption_note):
        self.raw: QuboInstance = raw                         # Original QuboInstance, unchanged
        self.source_convention: str = source_convention
        self.direction: str = direction            # Meaning ASSUMED for stored energy
        self.linear: dict = linear                  # {variable: polynomial coefficient}
        self.quadratic: dict = quadratic            # {(i, j): polynomial coefficient}
        self.offset:Decimal = offset                  # Constant term (Decimal)
        self.assumption_note: str = assumption_note
        self.num_variables: int = raw.num_variables


class ConvertedQubo:
    """
    The SAME objective, rewritten to a requested convention and direction.

    This is NOT a solver-object yet.
    'coefficients' is only a dictionary:
        {(i, j): stored_value}, with i <= j and 1-based indices.

    Treatment for off-diagonal values:
    If convention='symmetric', an off-diagonal stored_value represents
    BOTH Q[i,j] and Q[j,i] in x.T @ Q @ x (and counts twice).
    If convention='upper_triangular', it counts once.


    Treatments for optimization direction:
    If direction changed, the entire objective is negated (also the offset).
    The 'energy_sign' says how the new energy relates to the interpreted one:
        E_converted(x) = energy_sign * E_interpreted(x).

    """

    def __init__(self, interpreted, convention, direction,
                 coefficients, offset, energy_sign):
        """
            We hold the original interpretation of the raw QUBO (interpreted),
            Plus,
            We receive a 'requested convension' (convention, direction, more?).

            given an interpreted QUBO, This class helps to facilitate external requested interpretation.
            for example:
            MQLib solvers assume the off-diagonal values are symmetric, and the problem is to be maximized.
            Dwave solvers assume the off-diagonal values are upper-triangular and the problem is to be minimized.
            etc..
        """
        self.interpreted: InterpretedQubo = interpreted        # A trace to chosen assumptions
        self.num_variables: int = interpreted.num_variables
        self.convention = convention
        self.direction: str = direction
        self.coefficients = coefficients
        self.offset: Decimal = offset
        self.energy_sign = energy_sign


def _check_choice(value, allowed, name):
    """
    validates the choice of convention/direction/energy_sign is
    one of an allowed option:
    convernsion ["symmetric", "upper triangular"]
    direction ["minimization", " maximization"]
    etc..
    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")


def _decimal_offset(value):
    """
    Input: Decimal/int/decimal-string
    Output: finite Decimal.

    Handle an explicitly supplied offset without float rounding errors.
    """
    if isinstance(value, float):
        raise TypeError("Supply offsets as Decimal, int or string, not float")
    if not isinstance(value, (Decimal, int, str)) or isinstance(value, bool):
        raise TypeError("Offset must be Decimal, int or string")
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError(f"Invalid offset: {value!r}") from error
    if not result.is_finite():
        raise ValueError("Offset must be finite")
    return result


def interpret_qubo(raw: QuboInstance,
                   *,
                   source_convention,
                   source_direction,
                   offset_override=None,
                   assumption_note="") -> InterpretedQubo:
    """

    TL;DR : This is the first convention layer: a raw qubo is being attached to an interpretation.


    Input: raw QuboInstance + caller-selected assumptions.
    Output: InterpretedQubo with standard polynomial coefficients.
    Role: RAW -> INTERPRETED (no solver target involved).

    'symmetric': an off-diagonal stored entry counts twice (Qij and Qji).
    'upper_triangular': an off-diagonal stored entry counts once.
     (Diagonal entries always count once.)

    We treat the offset as an additive constant. Missing offsets are NOT
    silently assumed zero; choose offset_override=0 if that is your assumption.
    If supplied, offset_override deliberately takes precedence over the file.

    Important: source_direction is the caller's assumption about the stored
    energy; we do NOT copy the original model's '# maximize' comment.
    """
    if not isinstance(raw, QuboInstance):
        raise TypeError("raw must be a QuboInstance from load_qs()")
    _check_choice(source_convention, ("symmetric", "upper_triangular"), "source_convention")
    _check_choice(source_direction, ("minimize", "maximize"), "source_direction")

    if offset_override is None:
        if raw.offset is None:
            raise ValueError("Source offset missing; provide offset_override explicitly")
        offset = raw.offset
    else:
        offset = _decimal_offset(offset_override)

    # Only the off-diagonal coefficients depend on matrix convention.
    scale = 2 if source_convention == "symmetric" else 1
    linear = dict(raw.raw_diagonal)
    quadratic = {
        pair: value * scale
        for pair, value in raw.raw_off_diagonal.items()
    }

    return InterpretedQubo(
        raw=raw,
        source_convention=source_convention,
        direction=source_direction,
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        assumption_note=assumption_note,
    )


def convert_qubo(interpreted: InterpretedQubo,
                 *,
                 target_convention: str,
                 target_direction: str) -> ConvertedQubo:

    """
    Input: interpreted QUBO + explicit target convention and direction.
    Output: ConvertedQubo with a coefficient dictionary and offset.
    Role: INTERPRETED -> CONVENTION-CONVERTED .

    1) If the target's min/max direction differs, negate ALL terms and offset.
    2) If target='symmetric', divide pairwise polynomial coefficients by 2.
       If target='upper_triangular', leave those coefficients unchanged.
    3) Return a NEW representation; never modify the input or raw QUBO.

    """
    if not isinstance(interpreted, InterpretedQubo):
        raise TypeError("interpreted must be an InterpretedQubo")
    _check_choice(target_convention, ("symmetric", "upper_triangular"), "target_convention")
    _check_choice(target_direction, ("minimize", "maximize"), "target_direction")

    sign = 1 if interpreted.direction == target_direction else -1

    # All coefficients here are 1-based upper-half entries, regardless
    # of how a target will eventually serialize them.
    coefficients = {}
    for i, value in interpreted.linear.items():
        coefficients[(i, i)] = value * sign

    for pair, value in interpreted.quadratic.items():
        """
        Note:
        'interpreted.quadratic' : are already SCALED.
        if source convention == symmetric

        """
        if target_convention == "symmetric":
            coefficients[pair] = value * sign / Decimal(2)
        else:
            coefficients[pair] = value * sign

    return ConvertedQubo(
        interpreted=interpreted,
        convention=target_convention,
        direction=target_direction,
        coefficients=coefficients,
        offset=interpreted.offset * sign,
        energy_sign=sign,
    )


def evaluate_interpreted(interpreted, assignment):
    """Input: InterpretedQubo + 0/1 assignment (list/tuple or {1: bit, ...}).
    Output: Decimal objective value, INCLUDING offset.
    Role: check interpretation/transformations in tests and diagnostics.

    This is NOT the solver-facing 'return an f(x)' API. We will build that
    separately with the other solver-output adapters.
    """
    if not isinstance(interpreted, InterpretedQubo):
        raise TypeError("interpreted must be an InterpretedQubo")

    n = interpreted.num_variables
    if isinstance(assignment, dict):
        if set(assignment) != set(range(1, n + 1)):
            raise ValueError(f"Assignment must have exactly keys 1..{n}")
        bits = assignment
    else:
        if len(assignment) != n:
            raise ValueError(f"Assignment must have {n} bits")
        bits = {i: bit for i, bit in enumerate(assignment, start=1)}

    if any(type(bit) is not int or bit not in (0, 1) for bit in bits.values()):
        raise ValueError("Assignment values must be integer 0 or 1")

    total = interpreted.offset
    for i, coefficient in interpreted.linear.items():
        total += coefficient * bits[i]
    for (i, j), coefficient in interpreted.quadratic.items():
        total += coefficient * bits[i] * bits[j]
    return total
