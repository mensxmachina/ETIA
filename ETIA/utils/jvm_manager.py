import jpype
import os
import platform
import importlib
import importlib.resources
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def start_jvm(package='ETIA.CausalLearning.algorithms.jar_files'):
    """
    Starts the JVM with the JAR files from the specified package.

    Parameters:
        package (str): The Python package containing the JAR files.
    """
    if jpype.isJVMStarted():
        logger.info("JVM is already running.")
        return

    try:
        # Discover JAR files in the specified package
        jar_files = []
        try:
            jar_files = list(importlib.resources.files(package).glob('*.jar'))
        except AttributeError:
            # For Python versions < 3.9
            import pkgutil
            import glob
            package_path = importlib.util.find_spec(package).submodule_search_locations[0]
            jar_files = glob.glob(os.path.join(package_path, '*.jar'))

        if not jar_files:
            logger.error(f"No JAR files found in package '{package}'.")
            raise FileNotFoundError(f"No JAR files found in package '{package}'.")

        # Convert Path objects to absolute strings
        jar_paths = [str(jar.resolve()) for jar in jar_files]

        # Construct classpath based on OS
        system = platform.system()
        if system == 'Windows':
            classpath_sep = ';'
        else:
            classpath_sep = ':'

        classpath = classpath_sep.join(jar_paths)
        logger.debug(f"Constructed classpath: {classpath}")

        # Determine the path to the JVM library based on OS
        java_home = os.environ.get('JAVA_HOME')
        if not java_home:
            logger.error("JAVA_HOME environment variable is not set.")
            raise EnvironmentError("JAVA_HOME environment variable is not set.")

        if system == 'Windows':
            jvm_path = os.path.join(java_home, 'bin', 'server', 'jvm.dll')
        elif system == 'Darwin':  # macOS
            jvm_path = os.path.join(java_home, 'lib', 'server', 'libjvm.dylib')
        elif system == 'Linux':
            jvm_path = os.path.join(java_home, 'lib', 'server', 'libjvm.so')
        else:
            logger.error(f"Unsupported operating system: {system}")
            raise OSError(f"Unsupported operating system: {system}")

        if not os.path.exists(jvm_path):
            logger.error(f"JVM library not found at: {jvm_path}")
            raise FileNotFoundError(f"JVM library not found at: {jvm_path}")

        logger.debug(f"JVM Path: {jvm_path}")

        # Start the JVM
        jpype.startJVM(
            jvm_path,
            "-ea", classpath=
            f"{classpath}",
            convertStrings=False
        )
        logger.info("JVM started successfully.")

    except Exception as e:
        logger.error(f"Failed to start JVM: {e}")
        raise

def stop_jvm():
    """
    Shuts down the JVM if it is running.
    """
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
        logger.info("JVM shut down successfully.")
    else:
        logger.info("JVM is not running.")
'''
# Example Usage
def example():
    try:
        start_jvm()
        print(jpype.JPackage('org').apache.commons.lang3.tuple)
        # Your code that interacts with Java classes goes here

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        stop_jvm()
'''