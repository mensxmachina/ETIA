.PHONY: clean build install uninstall reinstall test

# Variables
PACKAGE_NAME = ETIA
PACKAGE_VERSION = 1.0.3
WHEEL_FILE = dist/$(PACKAGE_NAME)-$(PACKAGE_VERSION)-py3-none-any.whl
TEST_SCRIPT = test_autocd.py

all:
	rm -rf build dist $(PACKAGE_NAME).egg-info
	pip uninstall -y $(PACKAGE_NAME)
	python setup.py sdist bdist_wheel
	pip install $(WHEEL_FILE)

# Clean build directories
clean:
	rm -rf build dist $(PACKAGE_NAME).egg-info

# Build the package
build:
	python setup.py sdist bdist_wheel

# Uninstall the package
uninstall:
	pip uninstall -y $(PACKAGE_NAME)

# Install the package
install: build
	pip install $(WHEEL_FILE)

# Reinstall the package
reinstall: clean uninstall install

# Run the test script
test: install
	python $(TEST_SCRIPT)
