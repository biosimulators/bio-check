import toml


def change_project_name(pyproject_file, new_name):
    # Load the pyproject.toml file
    with open(pyproject_file, 'r') as file:
        pyproject_data = toml.load(file)

    # Change the project name
    if 'tool' in pyproject_data and 'poetry' in pyproject_data['tool']:
        pyproject_data['tool']['poetry']['name'] = new_name
    else:
        raise KeyError("Could not find the 'tool.poetry.name' key in pyproject.toml")

    # Save the updated pyproject.toml file
    with open(pyproject_file, 'w') as file:
        toml.dump(pyproject_data, file)

    print(f"Project name successfully changed to: {new_name}")


# Example usage
change_project_name('pyproject.toml', 'bio-compose')