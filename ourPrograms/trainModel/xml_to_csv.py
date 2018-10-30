import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import csv


def xml_to_csv(path):
    xml_list = []
    print(path)
    for xml_file in glob.glob(os.path.join(path, '*.xml')):

        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def remove_spaces(path):
    file = {}
    with open(path,  'r') as file:
        readie_boi = csv.reader(file, delimter=',')
        row = {}
        for column in readie_boi:
            col = ''
            for char in column:
                if char is not ' ':
                    col += char
            row.append(col)
        print(row)



def main():
    # Change this to neccesary path
    image_path = os.path.join(os.getcwd(), 'images', 'train')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(os.path.join('labels.csv'), index=None)
    remove_spaces(os.path.join('labels.csv'))

    print('Successfully converted xml to csv.')


main()