from urllib.request import urlopen
import re
from pprint import pprint
from xlrd import open_workbook

# img_data = requests.get('https://www.flickr.com/photos/92359315@N02/39960432520/').content
# with open('x.jpg', 'wb') as handler:
#     handler.write(img_data)
#
# # urllib.request.urlretrieve('https://www.flickr.com/photos/92359315@N02/39960432520/', 'tw.jpg')

wb = open_workbook('dataset.xls')
for sheet in wb.sheets():
    number_of_rows = sheet.nrows
    number_of_columns = sheet.ncols

    items = []

    rows = []
    col = number_of_columns-2
    for row in range(1, number_of_rows):
        im_id = (sheet.cell(row,0).value)
        url = (sheet.cell(row,col).value)
        class_label = (sheet.cell(row,number_of_columns-1).value)
        print(im_id, url)
        f = open('../data/'+str(class_label)+'/'+str(im_id)+'.jpg','wb')
        f.write(urlopen(url).read())
        f.close()
