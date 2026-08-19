from pydantic import BaseModel, Field
from typing import Literal

#custom exception classes with corresponding hints
class OutputVariableMissing(ValueError):
    hint = (
        "output_variable must exactly match one of the variable names in the variables list."
    )

class UndefinedConstraintReference(ValueError):
    hint = (
        "All variable names used in constraints (target_variable, operand_1, operand_2) "
        "must exactly match a name in the variables list. Check for typos or missing variables."
    )

class OrphanedNullVariable(ValueError):
    hint = (
        "Every variable with a null value must be the target of exactly one constraint. "
        "If the problem states its value directly, set value to that number instead of null."
    )

class SelfReferentialConstraint(ValueError):
    hint = (
        "Z3 solves all constraints simultaneously, not sequentially. "
        "You CANNOT accumulate into a variable across multiple steps. "
        "WRONG: total = total * days. "
        "RIGHT: introduce a new variable: total_with_days = total * days. "
        "Each variable must appear as target_variable in at most one constraint."
    )

#assumes there are not distractor variables, can alter this to say remove distractors if it includes them though later on with a bigger model
class UnusedVariable(ValueError):
    hint = (
        "You extracted a variable but never used it in any constraint. "
        "Every variable in the variables list must appear at least once as either "
        "operand_1, operand_2, or target_variable."
    )


class Variable(BaseModel):
    name: str = Field(description="Snake_case variable name.")
    value: float | None = Field(description="Numeric value. Null ONLY if it is the output variable.")
    unit: str | None = Field(description="The unit of measurement, if applicable.")

class Constraint(BaseModel):
    target_variable: str = Field(description="The variable being assigned (e.g., 'total_items')")
    operand_1: str | float = Field(description="A variable name or a hardcoded number")
    operator: Literal['+', '-', '*', '/'] = Field(description="The math operation to perform")
    operand_2: str | float = Field(description="A variable name or a hardcoded number")

class MathProblem(BaseModel):
    domain: str
    objective: str
    variables: list[Variable]
    constraints: list[Constraint]
    output_variable: str
    

def check_output_variable_exists(problem: MathProblem):
    defined = [v.name for v in problem.variables]
    if problem.output_variable not in defined:
        raise OutputVariableMissing(f"output_variable '{problem.output_variable}' not in variables.")

def check_constraint_references(problem: MathProblem):
    defined = {v.name for v in problem.variables}
    for c in problem.constraints:
        if c.target_variable not in defined:
            raise UndefinedConstraintReference(f"Constraint target '{c.target_variable}' is not a defined variable.")
        for field, val in [("operand_1", c.operand_1), ("operand_2", c.operand_2)]:
            if isinstance(val, str) and val not in defined:
                raise UndefinedConstraintReference(f"Constraint {field} '{val}' is not a defined variable.")

def check_no_orphaned_nulls(problem: MathProblem):
    computed = {c.target_variable for c in problem.constraints}
    for v in problem.variables:
        if v.value is None and v.name not in computed:
            raise OrphanedNullVariable(f"Variable '{v.name}' has value: null but is never assigned by any constraint.")

def check_no_self_referential_constraints(problem: MathProblem):
    for c in problem.constraints:
        if c.target_variable == c.operand_1 or c.target_variable == c.operand_2:
            raise SelfReferentialConstraint(
                f"You used '{c.target_variable}' as both the target_variable and an operand."
            )
    
def check_no_unused_variables(problem: MathProblem):
    # Gather all variables used in constraints
    used_in_constraints = set()
    for c in problem.constraints:
        used_in_constraints.add(c.target_variable)
        if isinstance(c.operand_1, str): used_in_constraints.add(c.operand_1)
        if isinstance(c.operand_2, str): used_in_constraints.add(c.operand_2)
    
    # Check if any extracted variable was left out
    for v in problem.variables:
        if v.name not in used_in_constraints:
            raise UnusedVariable(f"Variable '{v.name}' was defined but never used in any constraint.")
        

# Registry of all active rules
VALIDATORS = [
    check_output_variable_exists,
    check_constraint_references,
    check_no_orphaned_nulls,
    check_no_self_referential_constraints,
    check_no_unused_variables
]

def validate_math_logic(problem: MathProblem):
    """Runs all registered logical validation rules on a parsed MathProblem."""
    for rule in VALIDATORS:
        rule(problem)