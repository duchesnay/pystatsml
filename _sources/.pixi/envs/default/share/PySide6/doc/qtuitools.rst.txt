// @snippet quiloader-registercustomwidget
Registers a Python created custom widget to QUiLoader, so it can be recognized
when loading a `.ui` file. The custom widget type is passed via the
``customWidgetType`` argument. This is needed when you want to override a
virtual method of some widget in the interface, since duck punching will not
work with widgets created by QUiLoader based on the contents of the `.ui` file.

(Remember that
`duck punching virtual methods is an invitation for your own demise! <https://doc.qt.io/qtforpython/shiboken6/wordsofadvice.html#duck-punching-and-virtual-methods>`_)

Let's see an obvious example. If you want to create a new widget it's probable you'll end up
overriding :class:`~PySide6.QtGui.QWidget`'s :meth:`~PySide6.QtGui.QWidget.paintEvent` method.

.. code-block:: python

   class Circle(QWidget):
       def paintEvent(self, event):
           with QPainter(self) as painter:
               painter.setPen(self.pen)
               painter.setBrush(QBrush(self.color))
               painter.drawEllipse(event.rect().center(), 20, 20)

   # ...

   loader = QUiLoader()
   loader.registerCustomWidget(Circle)
   circle = loader.load('circle.ui')
   circle.show()

   # ...
// @snippet quiloader-registercustomwidget

// @snippet loaduitype
.. currentmodule:: PySide6.QtUiTools

loadUiType
***********
.. py:function:: loadUiType(uifile: str) -> tuple(object, object)

   :param str uifile: The name of the `.ui` file
   :return: tuple(object, object)

This function generates and loads a `.ui` file at runtime, and it returns
a `tuple` containing the reference to the Python class, and the base class.

We recommend not to use this approach as the workflow should be to generate a Python file
from the `.ui` file, and then import and load it to use it, but we do understand that
there are some corner cases when such functionality is required.

The internal process relies on `uic` being in the PATH.
The `pyside6-uic` wrapper uses a shipped `uic` that is located in the
`site-packages/PySide6/uic`, so PATH needs to be updated to use that if there
is no `uic` in the system.

A simple use case is::

    from PySide6.QtUiTools import loadUiType

    generated_class, base_class = loadUiType("themewidget.ui")
    # the values will be:
    #  (<class '__main__.Ui_ThemeWidgetForm'>, <class 'PySide6.QtWidgets.QWidget'>)

    widget = base_class()
    form = generated_class()
    form.setupUi(widget)
    # form.a_widget_member.a_method_of_member()
    widget.show()
// @snippet loaduitype
