// @snippet quick_test_main_documentation

Sets up the entry point for a Qt Quick Test application.
The ``name`` argument uniquely identifies this set of tests.

``sys.argv`` should be passed to the ``argv`` argument to ensure
propagation of the command line arguments.

.. note:: The function assumes that your test sources are in the current
          directory, unless the ``QUICK_TEST_SOURCE_DIR`` environment
          variable is set or a directory is passed in ``dir``.

The following snippet demonstrates the use of this function:

.. code-block:: Python

    import sys
    from PySide6.QtQuickTest import QUICK_TEST_MAIN

    ex = QUICK_TEST_MAIN("example", sys.argv)
    sys.exit(ex)


// @snippet quick_test_main_documentation

// @snippet quick_test_main_with_setup_documentation

Sets up the entry point for a Qt Quick Test application.
The ``name`` argument uniquely identifies this set of tests.

``sys.argv`` should be passed to the ``argv`` argument to ensure
propagation of the command line arguments.

This function is identical to ``QUICK_TEST_MAIN()``, except that it takes an
additional argument ``setup``, the type of a ``QObject``-derived
class which will be instantiated. With this class, it is possible to define
additional setup code to execute before running the QML test.

The following snippet demonstrates the use of this function:

.. code-block:: Python

    import sys
    from PySide6.QtQuickTest import QUICK_TEST_MAIN_WITH_SETUP

    class CustomTestSetup(QObject):
        def __init__(self, parent=None):
            super().__init__(parent)

        @Slot(QQmlEngine)
        def qmlEngineAvailable(self, qmlEngine):
            pass

    ex = QUICK_TEST_MAIN_WITH_SETUP("qquicktestsetup", CustomTestSetup, sys.argv)
    sys.exit(ex)


.. note:: The function assumes that your test sources are in the current
          directory, unless the ``QUICK_TEST_SOURCE_DIR`` environment
          variable is set or a directory is passed in ``dir``.

// @snippet quick_test_main_with_setup_documentation
