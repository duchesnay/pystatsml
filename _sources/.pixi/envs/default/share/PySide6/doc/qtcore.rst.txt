// @snippet q_arg
This function takes a type (or a type string) and a value of that type
and returns an internal object that can be passed to
:meth:`QMetaObject.invokeMethod`. See also Q_RETURN_ARG().
// @snippet q_arg

// @snippet q_return_arg
This macro takes a type (or a type string) a value of which is then
returned by :meth:`QMetaObject.invokeMethod`. See also Q_ARG().
// @snippet q_return_arg

// @snippet qlocale-system
Returns a QLocale object initialized to the system locale.

The system locale may use system-specific sources for locale data, where
available, otherwise falling back on QLocale's built-in database entry for the
language, script and territory the system reports.

For example, on Windows, this locale will use the decimal/grouping characters and
date/time formats specified in the system configuration panel.

.. note:: Qt for Python on macOS will not reflect the user's region and language
          preferences though QLocale::system(), but will instead reflect the
          environment variables POSIX uses to specify locale, similar to Python's
          locale module. If the system locale cannot be determined, which can be
          due to none of the variables 'LC_ALL', 'LC_CTYPE', 'LANG' or 'LANGUAGE'
          being set by your environment, then the default POSIX locale or
          'C' locale is returned.

See also c().
// @snippet qlocale-system

// @snippet qabstractitemmodel-createindex
Creates a model index for the given row and column with the internal pointer
ptr. When using a :class:`QSortFilterProxyModel`, its indexes have their own
internal pointer. It is not advisable to access this internal pointer outside
of the model. Use the ``data()`` function instead.

This function provides a consistent interface that model subclasses must use to
create model indexes.

.. warning:: Because of some Qt/Python integration rules, the ``ptr`` argument does
             not get the reference incremented during the QModelIndex life time.
             So it is necessary to keep the object used on ``ptr`` argument alive
             during the whole process. Do not destroy the object if you are not
             sure about that.
// @snippet qabstractitemmodel-createindex

// @snippet qobject-findChild
To find the child of a certain :class:`QObject`, the first argument of this
function should be the child's type, and the second the name of the child:

::

    ...
    parent = QWidget()
    ...
    # The first argument must be the child type
    child1 = parent.findChild(QPushButton, "child_button")
    child2 = parent.findChild(QWidget, "child_widget")

// @snippet qobject-findChild

// @snippet qcoreapplication-init
Constructs a Qt kernel application. Kernel applications are applications
without a graphical user interface. These type of applications are used
at the console or as server processes.

The *args* argument is processed by the application, and made available
in a more convenient form by the :meth:`~PySide6.QtCore.QCoreApplication.arguments()`
method.
// @snippet qcoreapplication-init

// @snippet qsettings-value
Custom overload that adds an optional named parameter to the function ``value()``
to automatically cast the type that is being returned by the function.

An example of this situation could be an ini file that contains
the value of a one-element list::

    settings.setValue('var', ['a'])

The the ini file will be::

    [General]
    var=a  # we cannot know that this is a list!

Once we read it, we could specify if we want
the default behavior, a str, or to cast the output
to a list.

    settings.value('var')  # Will get "a"
    settings.value('var', type=list)  # Will get ["a"]

// @snippet qsettings-value
