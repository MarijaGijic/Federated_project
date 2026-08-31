import xml.etree.ElementTree as ET


def _plist_values(dict_element):
    """Return key/value elements from an Apple plist <dict>."""
    children = list(dict_element)
    return {elem.text: children[i + 1] for i, elem in enumerate(children[:-1]) if elem.tag == "key"}


def _number(element, cast):
    if element is None or element.text is None:
        return None
    try:
        return cast(element.text)
    except (TypeError, ValueError):
        return None


def extract_rois_from_xml(xml_file):

    rois = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        dict_root = root.find('dict')
        images_array = dict_root.find('array')
        image_dict = images_array.find('dict')
        rois_array = image_dict.find('array')

        if rois_array is None:
            return rois
        
        roi_dicts = rois_array.findall('dict')

        for roi in roi_dicts:
            values = _plist_values(roi)
            points_px_array = values.get("Point_px")

            points = []
            if points_px_array is not None:
                for point in points_px_array.findall("string"):
                    x_str, y_str = point.text.strip("()").split(",")
                    x, y = int(round(float(x_str))), int(round(float(y_str)))
                    points.append((x, y))

            rois.append({
                "name": values.get("Name").text if values.get("Name") is not None else None,
                "type": _number(values.get("Type"), int),
                "area": _number(values.get("Area"), float),
                "number_of_points": _number(values.get("NumberOfPoints"), int),
                "index_in_image": _number(values.get("IndexInImage"), int),
                "points": points,
            })
            
        return rois
    
    except Exception as e:
        print(f"Failed to parse XML {xml_file}: {e}")
        return rois