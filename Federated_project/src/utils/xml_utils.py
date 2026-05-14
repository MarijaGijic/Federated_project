import xml.etree.ElementTree as ET

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
            children = list(roi)
            lesion_type = None
            points_px_array = None

            
            for i, elem in enumerate(children):
                if elem.tag == 'key' and elem.text == 'Name':
                    lesion_type = children[i + 1].text

            for i, elem in enumerate(children):
                if elem.tag == 'key' and elem.text == 'Point_px':
                    points_px_array = children[i + 1]
                    break
            
            if points_px_array is None:
                continue

            points = []
            for point in points_px_array.findall('string'):
                x_str, y_str = point.text.strip('()').split(',')
                x, y = int(round(float(x_str))), int(round(float(y_str)))
                points.append((x, y))
            
            if points:
                rois.append({
                    "lesion_type": lesion_type,
                    "points": points
                })
            
        return rois
    
    except Exception as e:
        print(f"Failed to parse XML {xml_file}: {e}")
        return rois