DEBUG · Coordinate info:

Latitude range: 40.000000 to 43.000000
Longitude range: -96.000000 to -90.038462
Non-null Latitude count: 24460
Non-null Longitude count: 24460
Valid coordinate pairs: 24460
DEBUG · Map bounds: (40.0, -96.0, 43.0, -90.03846153846153)

DEBUG · df_for_maps shape: (24460, 50)

DEBUG · Yield column exists: True

DEBUG · Yield range: 0.77 to 81.44

🔄 Rendering Yield (bu/ac) overlay...

✅ Yield (bu/ac) overlay rendered successfully

🔄 Rendering Variable Rate Profit ($/ac) overlay...

✅ Variable Rate Profit ($/ac) overlay rendered successfully

🔄 Rendering Fixed Rate Profit ($/ac) overlay...

✅ Fixed Rate Profit ($/ac) overlay rendered successfully

🗺️ Interactive map with yield data and overlays

DEBUG · Map object type:

classfolium.folium.Map(location: Optional[Sequence[float]] = None, width: Union[str, float] = '100%', height: Union[str, float] = '100%', left: Union[str, float] = '0%', top: Union[str, float] = '0%', position: str = 'relative', tiles: Union[str, folium.raster_layers.TileLayer, NoneType] = 'OpenStreetMap', attr: Optional[str] = None, min_zoom: Optional[int] = None, max_zoom: Optional[int] = None, zoom_start: int = 10, min_lat: float = -90, max_lat: float = 90, min_lon: float = -180, max_lon: float = 180, max_bounds: bool = False, crs: str = 'EPSG3857', control_scale: bool = False, prefer_canvas: bool = False, no_touch: bool = False, disable_3d: bool = False, png_enabled: bool = False, zoom_control: Union[bool, str] = True, font_size: str = '1rem', **kwargs: Union[str, float, bool, Sequence, dict, NoneType])
Create a Map with Folium and Leaflet.js

Generate a base map of given width and height with either default
tilesets or a custom tileset URL. Folium has built-in all tilesets
available in the ``xyzservices`` package. For example, you can pass
any of the following to the "tiles" keyword:

    - "OpenStreetMap"
    - "CartoDB Positron"
    - "CartoDB Voyager"

Explore more provider names available in ``xyzservices`` here:
https://leaflet-extras.github.io/leaflet-providers/preview/.

You can also pass a custom tileset by passing a
:class:`xyzservices.TileProvider` or a Leaflet-style
URL to the tiles parameter: ``https://{s}.yourtiles.com/{z}/{x}/{y}.png``.

Parameters
----------
location: tuple or list, default None
    Latitude and Longitude of Map (Northing, Easting).
width: pixel int or percentage string (default: '100%')
    Width of the map.
height: pixel int or percentage string (default: '100%')
    Height of the map.
tiles: str or TileLayer or :class:`xyzservices.TileProvider`, default 'OpenStreetMap'
    Map tileset to use. Can choose from a list of built-in tiles,
    pass a :class:`xyzservices.TileProvider`,
    pass a custom URL, pass a TileLayer object,
    or pass `None` to create a map without tiles.
    For more advanced tile layer options, use the `TileLayer` class.
min_zoom: int, optional, default 0
    Minimum allowed zoom level for the tile layer that is created.
    Filled by xyzservices by default.
max_zoom: int, optional, default 18
    Maximum allowed zoom level for the tile layer that is created.
    Filled by xyzservices by default.
zoom_start: int, default 10
    Initial zoom level for the map.
attr: string, default None
    Map tile attribution; only required if passing custom tile URL.
crs : str, default 'EPSG3857'
    Defines coordinate reference systems for projecting geographical points
    into pixel (screen) coordinates and back.
    You can use Leaflet's values :
    * EPSG3857 : The most common CRS for online maps, used by almost all
    free and commercial tile providers. Uses Spherical Mercator projection.
    Set in by default in Map's crs option.
    * EPSG4326 : A common CRS among GIS enthusiasts.
    Uses simple Equirectangular projection.
    * EPSG3395 : Rarely used by some commercial tile providers.
    Uses Elliptical Mercator projection.
    * Simple : A simple CRS that maps longitude and latitude into
    x and y directly. May be used for maps of flat surfaces
    (e.g. game maps). Note that the y axis should still be inverted
    (going from bottom to top).
control_scale : bool, default False
    Whether to add a control scale on the map.
prefer_canvas : bool, default False
    Forces Leaflet to use the Canvas back-end (if available) for
    vector layers instead of SVG. This can increase performance
    considerably in some cases (e.g. many thousands of circle
    markers on the map).
no_touch : bool, default False
    Forces Leaflet to not use touch events even if it detects them.
disable_3d : bool, default False
    Forces Leaflet to not use hardware-accelerated CSS 3D
    transforms for positioning (which may cause glitches in some
    rare environments) even if they're supported.
zoom_control : bool or position string, default True
    Display zoom controls on the map. The default `True` places it in the top left corner.
    Other options are 'topleft', 'topright', 'bottomleft' or 'bottomright'.
font_size : int or float or string (default: '1rem')
    The font size to use for Leaflet, can either be a number or a
    string ending in 'rem', 'em', or 'px'.
**kwargs
    Additional keyword arguments are passed to Leaflets Map class:
    https://leafletjs.com/reference.html#map

Returns
-------
Folium Map Object

Examples
--------
>>> m = folium.Map(location=[45.523, -122.675], width=750, height=500)
>>> m = folium.Map(location=[45.523, -122.675], tiles="cartodb positron")
>>> m = folium.Map(
...     location=[45.523, -122.675],
...     zoom_start=2,
...     tiles="https://api.mapbox.com/v4/mapbox.streets/{z}/{x}/{y}.png?access_token=mytoken",
...     attr="Mapbox attribution",
... )
default_csslist	[('leaflet_css', 'https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css'), ('bootstrap_css', 'https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css'), ('glyphicons_css', 'https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css'), ('awesome_markers_font_cs...
default_jslist	[('leaflet', 'https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js'), ('jquery', 'https://code.jquery.com/jquery-3.7.1.min.js'), ('bootstrap', 'https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js'), ('awesome_markers', 'https://cdnjs.cloudflare.com/ajax/libs/Leaflet.aw...
includesdict	{}
leaflet_class_nameproperty	Property attribute.
add_childfunction	Add a child.
add_childrenfunction	Add a child.
add_css_linkfunction	Add or update css resource link.
add_js_linkfunction	Add or update JS resource link.
add_tofunction	Add element to a parent.
fit_boundsfunction	Fit the map to contain a bounding box with the
get_boundsfunction	Computes the bounds of the object and all it's children
get_namefunction	Returns a string representation of the object.
get_rootfunction	Returns the root of the elements tree.
includemethod	No docs available
keep_in_frontfunction	Pass one or multiple layers that must stay in front.
onfunction	No docs available
oncefunction	No docs available
renderfunction	Renders the HTML representation of the element.
savefunction	Saves an Element into a file.
show_in_browserfunction	Display the Map in the default web browser.
to_dictfunction	Returns a dict representation of the object.
to_jsonfunction	Returns a JSON representation of the object.
DEBUG · Map location:

[
0:39.5
1:-98.35
]
DEBUG · Map zoom: 5

🔄 Creating optimized map...

✅ Optimized map created with caching to prevent flashing

🔄 Attempting Method 1: Basic st_folium...
