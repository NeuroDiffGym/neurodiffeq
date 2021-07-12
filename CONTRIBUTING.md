# Contribution Guidelines

These guidelines are *debatable*. If you have a better idea, feel free to discuss it.

## Rules of Thumb

1. **No two people should work on the same branch.** Only work on branches *you created*. To make changes to another contributor's branch, create a new branch (or a fork) and open a Pull Request to that person's branch.
2. **Always write unit tests and make sure they pass.** Unit tests in the `tests` folder are automatically run on [Travis-CI](https://travis-ci.org) when you push a commit or make a PR. Make sure they pass and aim for a high code coverage.
3. **Document your code.** Detail and complete documentation helps others, future contributors, and your future self understand what is going on.

## Style

### Python

We generally **follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style and naming convention**, with a maximum line width of 120 characters instead of 80. (Again, this is debatable. At least for me, an 80-char line wrap is too restrictive in the 2020s.)

**Exceptions may be made consistently when necessary. But PEP-8 style should be the first choice.**

Most IDEs (e.g. PyCharm) and editors (Vim, Sublime Text, VS Code etc.) have **extensions, plugins, or keyboard shortcuts to quickly format code** to honor PEP-8 with customizability.

### Git Commit Messages

To emphasize what a commit does, consider formatting commit messages as `action(target): message content`. 

### Action

Possible choices of `action`s are:

|    Action    | Usage                                                        |
| :----------: | :----------------------------------------------------------- |
|   **fix**    | Fix a bug. The bug should be resolved after the commit.      |
|    **to**    | Partially fix a bug. The bug is not completely resolved after the commit. |
| **enhance**  | Improve an existing functionality.                           |
|   **feat**   | Implement a new feature.                                     |
| **refactor** | Refactor the logic/structure of the code so that it's more SOLID or DRY. <br />*Renaming also counts as **refactor***. |
|  **style**   | Improve the style of the code for consistency.               |
|   **typo**   | Fix one or more typos.                                       |
|   **docs**   | Add or improve documentation.                                |
|   **test**   | Add or improve test cases.                                   |
|  **chore**   | Maintenance relate to the GitHub Repository, PyPI registry, etc. |
|   (others)   | If none above is applicable, pick another intuitive action name. |

### Target

Possible choices of `target` are:

1. Module/file names without extension, e.g., `solver`, `monitor`, `generator`, `callback`. Omit the `s` to save space.
2. Functionalities, e.g., `io`, `gui`. Use lowercase for consistency.
3. Miscellaneous, e.g., `pypi`, `hub`, `logic`.

Where appropriate, `target` can be omitted, but the parentheses and colon `():` should be retained.

### Message Content

There's no hard restrictions on message content, but we prefer starting a commit message with a verb in its original form (instead of its past tense) for the following reasons.

1. A *commit* message is not a *log* entry. Instead of *recollecting* what was done, it *calls for* changes to take place with itself in an imperative manner.
2. Commit message longer than 72 characters are wrapped on GitHub and almost all Git clients. Original form of verbs are typically shorter than the past tenses. (`do` vs `did`, `add` vs `added`).

See [this answer](https://stackoverflow.com/a/3580764/6685437) for more.

### Examples of Commit Messages

- `refactor(solver): implement a Solver2D`
- `fix(solution): fix a bug that always returns flattened numpy.ndarray`
- `test(callback): add test case for StopCallback`

## Development Environment Setup

To develop and debug locally, consider the following setup.

1. Clone the repository locally `git clone https://github.com/NeuroDiffGym/neurodiffeq.git && cd neurodiffeq`;
2. Install the library as editable source `pip install -e .`;
3. Create your own branch and swtich to it `git checkout -b <YOUR_BRANCH_NAME>`.

## Testing

We use [pytest](https://pytest.org) for unit testing.

For each source module `neurodiffeq/xxx.py`, there should be a test module `tests/test_xxx.py`, which should contain test cases for each function and class defined in the source module.

## Documentation

Documentation should be written for all public functions and classes in [RST](https://learnxinyminutes.com/docs/rst/) format in docstring. We use `sphinx.ext.autodocs` for  generating documentation. 

Before making PRs, make sure the docstrings are well formatted by previewing them locally:

1. Go to the documentation root: `cd <REPO_ROOT>/docs`.
2. Install requirements for generating documentation: `pip install -r requirements.txt`.
3. Generate static files `make html`. (Windows users may need to install `make` first).
4. Open `./_build/html/index.html` in browser and preview.
