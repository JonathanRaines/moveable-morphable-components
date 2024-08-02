def main() -> None:
    define_design_space()
    define_objective()
    define_constraints()
    initialise_components()

    for i in range(max_iterations):
        finite_element()
        sensitivity_analysis()
        update_design_variables()

        if is_converged():
            break


def define_design_space() -> None:
    raise NotImplementedError


def define_objective() -> None:
    raise NotImplementedError


def define_constraints() -> None:
    raise NotImplementedError

def initialise_components() -> None:
    raise NotImplementedError

def finite_element() -> None:
    raise NotImplementedError

def sensitivity_analysis() -> None:

def update_design_variables() -> None:
    raise NotImplementedError

def is_converged() -> bool:
    raise NotImplementedError

if __name__ == "__main__":
    main()
