'''74aa621e9062af8a580dc92922151dda'''
'''2e65d83a0df0bd28'''
import flickrapi
from pprint import pprint
import json
import xlwt

api_key = u'74aa621e9062af8a580dc92922151dda'
api_secret = u'2e65d83a0df0bd28'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
emotions = ['sarcasm', 'sarcastic', 'irony', 'wit', 'satire', 'non-sarcastic', 'nonsarcastic', 'compliment', 'fact', 'information', 'praise', 'applause']
all_photo_info = {}
for emo in emotions:
    photos = flickr.photos.search(tags=emo, tag_mode = 'any', sort='interestingness-desc')['photos']
    # print(photos)
    page_to_get = 101
    if(page_to_get > int(photos['pages'])):
        page_to_get = int(photos['pages'])
    ids = []
    for p in range(1, page_to_get):
        photos = flickr.photos.search(tags=emo, tag_mode = 'any', sort='interestingness-desc', page=p, license=[1, 2, 3, 4, 5, 6])['photos']
        photos = photos['photo']
        # pprint(photos)
        for seq in range(len(photos)):
            ids.append(photos[seq]['id'])

    all_photo_info[emo] = ids
    # break

# pprint(all_photo_info)
print('hi')
count = 0
for key in all_photo_info.keys():
    print(len(all_photo_info[key]))
    count += len(all_photo_info[key])

print(count)
with open('file.txt', 'w') as file:
     file.write(json.dumps(all_photo_info))

dataset = {'1':[], '0':[]}
p_id_set = set()
n_id_set = set()
for key in all_photo_info.keys():
    if key in ['sarcasm', 'sarcastic', 'irony', 'wit', 'satire']:
        print(key)
        count = 0
        for img_id in all_photo_info[key]:
            if img_id not in p_id_set:
                p_id_set.add(img_id)
                dataset['1'].append(img_id)
                count += 1
        print(count)
    elif key in ['non-sarcastic', 'nonsarcastic', 'compliment', 'fact', 'information', 'praise', 'applause']:
        print(key)
        count = 0
        for img_id in all_photo_info[key]:
            if img_id in p_id_set:
                p_id_set.remove(img_id)
                n_id_set.add(img_id)
            if img_id not in n_id_set:
                n_id_set.add(img_id)
                dataset['0'].append(img_id)
                count += 1
        print(count)
#     for img_id in all_photo_info[key]:
#         if img_id not in id_set:
#             id_set.add(img_id)
#             elif key in ['non-sarcastic', 'nonsarcastic', 'compliment', 'fact', 'information', 'praise', 'applause']:
#                 dataset['0'].append(img_id)
#
# pprint(dataset)
# print(dataset.keys())
print('1', len(dataset['1']))
print('0', len(dataset['0']))

columns = ['id', 'title', 'description', 'tags', 'url', 'class']

book = xlwt.Workbook()
for class_label in dataset.keys():
    sheet = book.add_sheet(class_label)
    for index, col_header in enumerate(columns):
        sheet.row(0).write(index, col_header)
    class_photo_info = dataset[class_label]
    for image in range(len(class_photo_info)):
        row = sheet.row(image+1)
        image_id = class_photo_info[image]
        image_info = flickr.photos.getInfo(photo_id=image_id)['photo']
        title = image_info['title']['_content']
        description = image_info['description']['_content']
        x_tags = ''
        for t in range(len(image_info['tags']['tag'])):
            x_tags += image_info['tags']['tag'][t]['raw']+ ' '
        tags = x_tags
        image_url = flickr.photos.getSizes(photo_id=image_id)['sizes']['size'][-1]['source']
        for index, col_header in enumerate(columns):
            if col_header == 'id':
                value = image_id
            elif col_header == 'title':
                value = title
            elif col_header == 'description':
                value = description
            elif col_header == 'tags':
                value = tags
            elif col_header == 'url':
                value = image_url
            elif col_header == 'class':
                value = class_label
            try:
                row.write(index, value)
            except Exception as e:
                print(e)

book.save('dataset.xls')


# x = flickr.photos.getSizes(photo_id='27995593008')['sizes']['size'][-1]['source']
# # pprint(x)
# x = flickr.photos.getInfo(photo_id='27995593008')['photo']
# # pprint(x['description']['_content'])
# # pprint(x['title']['_content'])
# print(x['comments'])
# x_tags = ''
# for t in range(len(x['tags']['tag'])):
#     x_tags += x['tags']['tag'][t]['raw']+ ' '
# print(x['tags']['tag'])
# print(x_tags)
# page_count = photos['photo']
# # sets = flickr.photosets.getList()
# print(photos)
# print(len(page_count))

# for i in range(len(sets['photosets']['photoset'])):
#     title  = sets['photosets']['photoset'][i]['title']['_content']
#     description = sets['photosets']['photoset'][i]['description']['_content']
#     print(title, description)



