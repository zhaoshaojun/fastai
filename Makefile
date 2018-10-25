# usage: make help

# notes:
# 'target: | target1 target2' syntax enforces the exact order

.PHONY: clean clean-test clean-pyc clean-build docs help clean-pypi clean-build-pypi clean-pyc-pypi clean-test-pypi dist-pypi upload-pypi clean-conda clean-build-conda clean-pyc-conda clean-test-conda dist-conda upload-conda test tag bump bump-minor bump-major bump-dev bump-minor-dev bump-major-dev commit-tag git-pull git-not-dirty test-install dist-pypi-bdist dist-pypi-sdist upload release

version_file = fastai/version.py
version = $(shell python setup.py --version)
cur_branch = $(shell git branch | sed -n '/\* /s///p')

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

define WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH =
# note that when:
# bash -c "command" arg1
# is called, the first argument is actually $0 and not $1 as it's inside bash!
#
# is_pip_ver_available "1.0.14"
# returns (echo's) 1 if yes, 0 otherwise
#
# since pip doesn't have a way to check whether a certain version is available,
# here we are using a hack, calling:
# pip install fastai==
# which doesn't find the unspecified version and returns all available
# versions instead, which is what we search
function is_pip_ver_available() {
    local ver="$$0"
    local out="$$(pip install fastai== |& grep $$ver)"
    if [[ -n "$$out" ]]; then
        echo 1
    else
        echo 0
    fi
}

function wait_till_pip_ver_is_available() {
    local ver="$$1"
    if [[ $$(is_pip_ver_available $$ver) == "1" ]]; then
        echo "fastai-$$ver is available on pypi"
        return 0
    fi

    COUNTER=0
    echo "waiting for fastai-$$ver package to become visible on pypi:"
    while [[ $$(is_pip_ver_available $$ver) != "1" ]]; do
        echo -en "\\rwaiting: $$COUNTER secs"
        COUNTER=$$[$$COUNTER +5]
	    sleep 5
    done
    echo -e "\rwaited: $$COUNTER secs    "
    echo -e "fastai-$$ver is now available on pypi"
}

echo "checking version $$0"
wait_till_pip_ver_is_available "$$0"
endef
export WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ PyPI

clean-pypi: clean-build-pypi clean-pyc-pypi clean-test-pypi ## remove all build, test, coverage and python artifacts

clean-build-pypi: ## remove pypi build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc-pypi: ## remove python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test-pypi: ## remove pypi test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist-pypi-bdist: ## build pypi wheel package
	@echo "\n\n*** Building pypi wheel package"
	python setup.py bdist_wheel

dist-pypi-sdist: ## build pypi source package
	@echo "\n\n*** Building pypi source package"
	python setup.py sdist

dist-pypi: | clean-pypi dist-pypi-sdist dist-pypi-bdist ## build pypi source and wheel package
	ls -l dist

upload-pypi: ## upload pypi package
	@echo "\n\n*** Uploading" dist/* "to pypi\n"
	twine upload dist/*


##@ Conda

clean-conda: clean-build-conda clean-pyc-conda clean-test-conda ## remove all build, test, coverage and python artifacts

clean-build-conda: ## remove conda build artifacts
	@echo "\n\n*** conda build purge"
	conda build purge-all
	@echo "\n\n*** rm -fr conda-dist/"
	rm -fr conda-dist/

clean-pyc-conda: ## remove conda python file artifacts

clean-test-conda: ## remove conda test and coverage artifacts

dist-conda: | clean-conda dist-pypi-sdist ## build conda package
	@echo "\n\n*** Building conda package"
	mkdir "conda-dist"
	conda-build ./conda/ -c pytorch -c fastai/label/main --output-folder conda-dist
	ls -l conda-dist/noarch/*tar.bz2

upload-conda: ## upload conda package
	@echo "\n\n*** Uploading" conda-dist/noarch/*tar.bz2 "to fastai@anaconda.org\n"
	anaconda upload conda-dist/noarch/*tar.bz2 -u fastai



##@ Combined (pip and conda)

# currently, no longer needed as we now rely on sdist's tarball for conda source, which doesn't have any data in it already
# find ./data -type d -and -not -regex "^./data$$" -prune -exec rm -rf {} \;

clean: clean-pypi clean-conda ## clean pip && conda package

dist: clean dist-pypi dist-conda ## build pip && conda package

upload: upload-pypi upload-conda ## upload pip && conda package

install: clean ## install the package to the active python's site-packages
	python setup.py install

test: ## run tests with the default python
	python setup.py --quiet test

tools-update: ## install/update build tools
	@echo "\n\n*** Updating build tools"
	conda install -y conda-verify conda-build anaconda-client
	pip install -U twine

release: ## do it all (other than testing)
	${MAKE} tools-update
	${MAKE} master-branch-switch
	${MAKE} git-not-dirty
	${MAKE} bump
	${MAKE} changes-finalize
	${MAKE} release-branch-create
	${MAKE} commit-version
	${MAKE} master-branch-switch
	${MAKE} bump-dev
	${MAKE} changes-dev-cycle
	${MAKE} commit-dev-cycle-push
	${MAKE} prev-branch-switch
	${MAKE} test
	${MAKE} commit-tag-push
	${MAKE} dist
	${MAKE} upload
	${MAKE} test-install
	${MAKE} backport-check
	${MAKE} master-branch-switch

##@ git helpers

git-pull: ## git pull
	@echo "\n\n*** Making sure we have the latest checkout"
	git checkout master
	git pull
	git status

git-not-dirty:
	@echo "*** Checking that everything is committed"
	@if [ -n "$(shell git status -s)" ]; then\
		echo "git status is not clean. You have uncommitted git files";\
		exit 1;\
	else\
		echo "git status is clean";\
    fi

prev-branch-switch:
	@echo "\n\n*** [$(cur_branch)] Switching to prev branch"
	git checkout -
	$(eval branch := $(shell git branch | sed -n '/\* /s///p'))
	@echo "Now on [$(branch)] branch"

release-branch-create:
	@echo "\n\n*** [$(cur_branch)] Creating release-$(version) branch"
	git checkout -b release-$(version)
	$(eval branch := $(shell git branch | sed -n '/\* /s///p'))
	@echo "Now on [$(branch)] branch"

release-branch-switch:
	@echo "\n\n*** [$(cur_branch)] Switching to release-$(version) branch"
	git checkout release-$(version)
	$(eval branch := $(shell git branch | sed -n '/\* /s///p'))
	@echo "Now on [$(branch)] branch"

master-branch-switch:
	@echo "\n\n*** [$(cur_branch)] Switching to master branch: version $(version)"
	git checkout master
	$(eval branch := $(shell git branch | sed -n '/\* /s///p'))
	@echo "Now on [$(branch)] branch"

commit-dev-cycle-push: ## commit version and CHANGES and push
	@echo "\n\n*** [$(cur_branch)] Start new dev cycle: $(version)"
	git commit -m "new dev cycle: $(version)" $(version_file) CHANGES.md

	@echo "\n\n*** [$(cur_branch)] Push changes"
	git push

commit-version: ## commit and tag the release
	@echo "\n\n*** [$(cur_branch)] Start release branch: $(version)"
	git commit -m "starting release branch: $(version)" $(version_file)
	$(eval branch := $(shell git branch | sed -n '/\* /s///p'))
	@echo "Now on [$(branch)] branch"

commit-tag-push: ## commit and tag the release
	@echo "\n\n*** [$(cur_branch)] Commit CHANGES.md"
	git commit -m "version $(version) release" CHANGES.md || echo "no changes to commit"

	@echo "\n\n*** [$(cur_branch)] Tag $(version) version"
	git tag -a $(version) -m "$(version)" && git push --tags

	@echo "\n\n*** [$(cur_branch)] Push changes"
	git push --set-upstream origin release-$(version)

# check whether there any commits besides fastai/version.py and CHANGES.md
# from the point of branching of release-$(version) till its HEAD. If
# there are any, then most likely there are things to backport.
backport-check: ## backport to master check
	@echo "\n\n*** [$(cur_branch)] Checking if anything needs to be backported"
	$(eval start_rev := $(shell git rev-parse --short $$(git merge-base master origin/release-$(version))))
	@if [ ! -n "$(start_rev)" ]; then\
		echo "*** failed, check you're on the correct release branch";\
		exit 1;\
	fi
	$(eval log := $(shell git log --oneline $(start_rev)..origin/release-$(version) -- . ":(exclude)fastai/version.py" ":(exclude)CHANGES.md"))
	@if [ -n "$(log)" ]; then\
		echo "!!! These commits may need to be backported:\n\n$(log)\n\nuse 'git show <commit>' to review or go to https://github.com/fastai/fastai/compare/release-$(version) to do it visually\nFor backporting see: https://docs-dev.fast.ai/release#backporting-release-branch-to-master";\
	else\
		echo "Nothing to backport";\
    fi


##@ Testing new package installation

test-install: ## test conda/pip package by installing that version them
	@echo "\n\n*** [$(cur_branch)] Branch check (needing release branch)"
	@if [ "$(cur_branch)" = "master" ]; then\
		echo "Error: you are not on the release branch, to switch to it do:\n  git checkout release-1.0.??\nafter adjusting the version number. Also possible that:\n  git checkout - \nwill do the trick, if you just switched from it. And then repeat:\n  make test-install\n";\
		exit 1;\
	else\
		echo "You're on the release branch, good";\
	fi

	@echo "\n\n*** Install/uninstall $(version) pip version"
	@pip uninstall -y fastai

	@echo "\n\n*** waiting for $(version) pip version to become visible"
	bash -c "$$WAIT_TILL_PIP_VER_IS_AVAILABLE_BASH" $(version)

	pip install fastai==$(version)
	pip uninstall -y fastai

	@echo "\n\n*** Install/uninstall $(version) conda version"
	@# skip, throws error when uninstalled @conda uninstall -y fastai

	@echo "\n\n*** waiting for $(version) conda version to become visible"
	@perl -e '$$v=shift; $$p="fastai"; $$|++; sub ok {`conda search -c fastai $$p==$$v 2>1 >/dev/null`; return $$? ? 0 : 1}; print "waiting for $$p-$$v to become available on conda\n"; $$c=0; while (not ok()) { print "\rwaiting: $$c secs"; $$c+=5;sleep 5; }; print "\n$$p-$$v is now available on conda\n"' $(version)

	conda install -y -c fastai fastai==$(version)
	@# leave conda package installed: conda uninstall -y fastai


##@ CHANGES.md file targets

changes-finalize: ## fix the version and stamp the date
	@echo "\n\n*** [$(cur_branch)] Adjust '## version (date)' in CHANGES.md"
	perl -pi -e 'use POSIX qw(strftime); BEGIN{$$date=strftime "%Y-%m-%d", localtime};s|^##.*Work In Progress\)|## $(version) ($$date)|' CHANGES.md

changes-dev-cycle: ## insert new template + version
	@echo "\n\n*** [$(cur_branch)] Install new template + version in CHANGES.md"
	perl -0777 -pi -e 's|^(##)|\n\n## $(version) (Work In Progress)\n\n### New:\n\n### Changed:\n\n### Fixed:\n\n\n\n$$1|ms' CHANGES.md


##@ Version bumping

# Support semver, but using python's .dev0 instead of -dev0

bump-patch: ## bump patch-level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2, $$3, $$4+1); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump: bump-patch ## alias to bump-patch (as it's used often)

bump-minor: ## bump minor-level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2, $$3+1, $$4); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-major: ## bump major-level unless has .devX, then don't bump, but remove .devX
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=$$5 ? join(".", $$2, $$3, $$4) :join(".", $$2+1, $$3, $$4); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-patch-dev: ## bump patch-level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3, $$4+1, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-dev: bump-patch-dev ## alias to bump-patch-dev (as it's used often)

bump-minor-dev: ## bump minor-level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2, $$3+1, $$4, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)

bump-major-dev: ## bump major-level and add .dev0
	@perl -pi -e 's|((\d+)\.(\d+).(\d+)(\.\w+\d+)?)|$$o=$$1; $$n=join(".", $$2+1, $$3, $$4, "dev0"); print STDERR "\n\n*** [$(cur_branch)] Changing version: $$o => $$n\n"; $$n |e' $(version_file)


##@ Coverage

coverage: ## check code coverage quickly with the default python
	coverage run --source fastai -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html
