# Testing fastai

## Notebook integration tests

Currently we have few automated tests, so most testing is through integration tests done in Jupyter Notebooks. The two places you should check for notebooks to test your code with are:

 - [The fastai examples](https://github.com/fastai/fastai/tree/master/examples)
 - [The fastai_docs notebooks](https://github.com/fastai/fastai_docs/tree/master/docs_src)

In each case, look for notebooks that have names starting with the application you're working on - e.g. 'text' or 'vision'.


## Automated tests

At the moment there are only a few automated tests, so we need to start expanding it! It's not easy to properly automatically test ML code, but there's lots of opportunities for unit tests.

We use [pytest](https://docs.pytest.org/en/latest/). Here is a complete [pytest API reference](https://docs.pytest.org/en/latest/reference.html).

The tests have been configured to automatically run against the git checked out `fastai` repository and not pre-installed `fastai`.

### Choosing which tests to run

To run all the tests:


   ```
   make test
   ```

or


   ```
   python setup.py test
   ```

To run an individual test file:

   ```
   pytest tests/test_core.py
   ```

Run tests by keyword expressions:

   ```
   pytest -k "list and not listify" tests/test_core.py
   ```

For example, if we have the following tests:

   ```
   def test_whatever():
   def test_listify(p, q, expected):
   def test_listy():
   ```

it will first select `test_listify` and `test_listy`, and then deselect `test_listify`, resulting in only the sub-test `test_listy` being run.

More ways: https://docs.pytest.org/en/latest/usage.html

For nuances of configuring pytest's repo-wide behavior see [collection](https://docs.pytest.org/en/latest/example/pythoncollection.html).



### To GPU or not to GPU


On a GPU-enabled setup, to test in CPU-only mode add `CUDA_VISIBLE_DEVICES=""`:
   ```
   CUDA_VISIBLE_DEVICES="" pytest tests/test_vision.py
   ```

To do the same inside the code of the test:
   ```
   fastai.torch_core.default_device = torch.device('cpu')
   ```

To switch back to cuda:
   ```
   fastai.torch_core.default_device = torch.device('cuda')
   ```

Make sure you don't hard-code any specific device ids in the test, since different users may have a different GPU setup. So avoid code like:
   ```
   fastai.torch_core.default_device = torch.device('cuda:1')
   ```
which tells `torch` to use the 2nd GPU. Instead, if you'd like to run a test locally on a different GPU, use the `CUDA_VISIBLE_DEVICES` environment variable:
   ```
   CUDA_VISIBLE_DEVICES="1" pytest tests/test_vision.py
   ```



### Report each sub-test name and its progress

For a single or a group of tests via `pytest`:

   ```
   pytest --pspec tests/test_fastai.py
   pytest --pspec tests
   ```

For all tests via `setup.py`:

   ```
   python setup.py test --addopts="--pspec"
   ```

### Output capture

During test execution any output sent to `stdout` and `stderr` is captured. If a test or a setup method fails, its according captured output will usually be shown along with the failure traceback.

To disable capturing and get the output normally use `-s` or `--capture=no`:

   ```
   pytest -s tests/test_core.py
   ```


### Color control

To have no color (e.g. yellow on white bg is not readable):

   ```
   pytest --color=no tests/test_core.py
   ```



### Sending test report to online pastebin service

Creating a URL for each test failure:

   ```
   pytest --pastebin=failed tests/test_core.py
   ```

This will submit test run information to a remote Paste service and provide a URL for each failure. You may select tests as usual or add for example -x if you only want to send one particular failure.

Creating a URL for a whole test session log:

   ```
   pytest --pastebin=all tests/test_core.py
   ```



# Writing Tests

XXX: Needs to be written
