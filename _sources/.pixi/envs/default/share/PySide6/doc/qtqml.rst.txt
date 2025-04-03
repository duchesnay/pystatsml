// @snippet qmlregistersingletoninstance
.. py:function:: qmlRegisterSingletonInstance(pytype: type,\
                                              uri: str,\
                                              versionMajor: int,\
                                              versionMinor: int,\
                                              typeName: str,\
                                              instanceObject: object) -> int

   :param type pytype: Python class
   :param str uri: uri to use while importing the component in QML
   :param int versionMajor: major version
   :param int versionMinor: minor version
   :param str typeName: name exposed to QML
   :param object instanceObject: singleton object to be registered
   :return: int (the QML type id)

This function registers a singleton Python object *instanceObject*, with a
particular *uri* and *typeName*. Its version is a combination of *versionMajor*
and *versionMinor*. Use this function to register an object of the given type
*pytype* as a singleton type.
// @snippet qmlregistersingletoninstance

// @snippet qmlregistersingletontype_qobject_nocallback
.. py:function:: qmlRegisterSingletonType(pytype: type, uri: str, versionMajor: int, versionMinor: int, typeName: str) -> int

   :param type pytype: Python class
   :param str uri: uri to use while importing the component in QML
   :param int versionMajor: major version
   :param int versionMinor: minor version
   :param str typeName: name exposed to QML
   :return: int (the QML type id)

This function registers a Python type as a singleton in the QML system.

Alternatively, the :ref:`QmlSingleton` decorator can be used.
// @snippet qmlregistersingletontype_qobject_nocallback

// @snippet qmlregistersingletontype_qobject_callback
.. py:function:: qmlRegisterSingletonType(pytype: type, uri: str, versionMajor: int, versionMinor: int, typeName: str, callback: object) -> int

   :param type pytype: Python class
   :param str uri: uri to use while importing the component in QML
   :param int versionMajor: major version
   :param int versionMinor: minor version
   :param str typeName: name exposed to QML
   :param object callback: Python callable (to handle Python type)
   :return: int (the QML type id)

This function registers a Python type as a singleton in the QML system using
the provided callback (which gets a QQmlEngine as a parameter) to generate the
singleton.
// @snippet qmlregistersingletontype_qobject_callback

// @snippet qmlregistersingletontype_qjsvalue
.. py:function:: qmlRegisterSingletonType(uri: str, versionMajor: int, versionMinor: int, typeName: str, callback: object) -> int

   :param str uri: uri to use while importing the component in QML
   :param int versionMajor: major version
   :param int versionMinor: minor version
   :param str typeName: name exposed to QML
   :param object callback: Python callable (to handle QJSValue)
   :return: int (the QML type id)

This function registers a QJSValue as a singleton in the QML system using the
provided callback (which gets a QQmlEngine as a parameter) to generate the
singleton.
// @snippet qmlregistersingletontype_qjsvalue

// @snippet qmlregistertype
.. py:function:: qmlRegisterType(pytype: type, uri: str, versionMajor: int, versionMinor: int, qmlName: str) -> int

   :param type pytype: Python class
   :param str uri: uri to use while importing the component in QML
   :param int versionMajor: major version
   :param int versionMinor: minor version
   :param str qmlName: name exposed to QML
   :return: int (the QML type id)

This function registers the Python *type* in the QML system with the name
*qmlName*, in the library imported from *uri* having the version number
composed from *versionMajor* and *versionMinor*. For example, this registers a
Python class 'MySliderItem' as a QML type named 'Slider' for version '1.0' of a
module called 'com.mycompany.qmlcomponents':

   ::

       qmlRegisterType(MySliderItem, "com.mycompany.qmlcomponents", 1, 0, "Slider")

Once this is registered, the type can be used in QML by importing the specified
module name and version number:

   ::

       import com.mycompany.qmlcomponents 1.0

       Slider { ... }

Note that it's perfectly reasonable for a library to register types to older
versions than the actual version of the library. Indeed, it is normal for the
new library to allow QML written to previous versions to continue to work, even
if more advanced versions of some of its types are available.
// @snippet qmlregistertype

// @snippet qmlregisteruncreatabletype
.. py:function:: qmlRegisterUncreatableType(pytype: type, uri: str, versionMajor: int, versionMinor: int, qmlName: str, noCreationReason: str) -> int

   :param type pytype: Python class
   :param str uri: uri to use while importing the component in QML
   :param int versionMajor: major version
   :param int versionMinor: minor version
   :param str qmlName: name exposed to QML
   :param str noCreationReason: Error message shown when trying to create the QML type
   :return: int (the QML type id)

This function registers the Python *type* in the QML system as an uncreatable
type with the name *qmlName*, in the library imported from *uri* having the
version number composed from *versionMajor* and *versionMinor*, showing
*noCreationReason* as an error message when creating the type is attempted. For
example, this registers a Python class 'MySliderItem' as a QML type named
'Slider' for version '1.0' of a module called 'com.mycompany.qmlcomponents':

   ::
       qmlRegisterUncreatableType(MySliderItem, "com.mycompany.qmlcomponents", 1, 0, "Slider", "Slider cannot be created.")

Note that it's perfectly reasonable for a library to register types to older
versions than the actual version of the library. Indeed, it is normal for the
new library to allow QML written to previous versions to continue to work, even
if more advanced versions of some of its types are available.

Alternatively, the :ref:`QmlUncreatable` decorator can be used.
// @snippet qmlregisteruncreatabletype

// @snippet qqmlengine-singletoninstance-qmltypeid
Returns the instance of a singleton type that was registered under qmlTypeId.
For ``QObject``-derived singleton types, the ``QObject`` instance is returned,
otherwise a ``QJSValue`` or ``None``.

It is recommended to store the QML type id, e.g. as a static member in the
singleton class. The lookup via qmlTypeId() is costly.
// @snippet qqmlengine-singletoninstance-qmltypeid

// @snippet qqmlengine-singletoninstance-typename
Returns the instance of a singleton type named typeName from the module specified
by uri. For ``QObject``-derived singleton types, the ``QObject`` instance is
returned, otherwise a ``QJSValue`` or ``None``.

This method can be used as an alternative to calling qmlTypeId followed by the
id based overload of singletonInstance. This is convenient when one only needs
to do a one time setup of a singleton; if repeated access to the singleton is
required, caching its typeId will allow faster subsequent access via the
type-id based overload.
// @snippet qqmlengine-singletoninstance-typename
