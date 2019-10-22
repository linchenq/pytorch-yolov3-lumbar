FOLDER = 'GT30'
DATABASE = 'The Lumbar GT30 Database'
ANNOTATION = 'LUMBAR GT30'

from lxml import etree

class Xmlgenerator(object):
    def __init__(self, filename, size, o_dict):
        '''
        Parameters Defination:
            filename : str, the name of the image with suffix
            size: (height, width, depth/channel), tuple
            o_dict : { name : (xmin, ymin, xmax, ymax) }

        Important notes:
            xmin, ..., ymax SHOULD be 1-based pixels
        '''
        super().__init__()
        self.filename = filename
        self.height, self.width, self.depth = size
        self.o_dict = o_dict

        self.root = self._generate()

    def output(self, path=None):
        s = etree.tostring(self.root, pretty_print=True)
        if path is not None:
            with open(path, 'wb') as f:
                f.write(s)
        else:
            print(s)

    def _generate(self):
        # create xml
        root = etree.Element('annotation')
        root.append(self._generate_child('folder', FOLDER))
        root.append(self._generate_child('filename', self.filename))

        # source
        source = etree.Element('source')
        root.append(source)
        source.append(self._generate_child('database', DATABASE))
        source.append(self._generate_child('annotation', ANNOTATION))

        # size
        t_size = etree.Element('size')
        root.append(t_size)
        t_size.append(self._generate_child('width', str(self.width)))
        t_size.append(self._generate_child('height', str(self.height)))
        t_size.append(self._generate_child('depth', str(self.depth)))

        # object
        t_object = etree.Element('object')
        root.append(t_object)
        for k, v in self.o_dict.items():
            t_object.append(self._generate_child('name', k))

            #bnd box
            bndbox = etree.Element('bndbox')
            t_object.append(bndbox)

            bndbox.append(self._generate_child('xmin', str(v[0])))
            bndbox.append(self._generate_child('ymin', str(v[1])))
            bndbox.append(self._generate_child('xmax', str(v[2])))
            bndbox.append(self._generate_child('ymax', str(v[3])))

        return root



    def _generate_child(self, prefix, context=None):
        child = etree.Element(prefix)
        if context is not None:
            child.text = context
        return child

if __name__ == '__main__':
    test_dict = {
            'person': (1, 3, 7, 9),
            'dog': (67, 98, 102, 187)
    }
    gen = Xmlgenerator('test.jpg', (256, 128, 3), test_dict)
    gen.output()
    gen.output(path='testoutput.txt')

'''
<annotation>
	<folder>VOC2007</folder>
	<filename>000001.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<owner>
		<flickrid>Fried Camels</flickrid>
		<name>Jinky the Fruit Bat</name>
	</owner>
	<size>
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>dog</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>8</xmin>
			<ymin>12</ymin>
			<xmax>352</xmax>
			<ymax>498</ymax>
		</bndbox>
	</object>
</annotation>
'''
















