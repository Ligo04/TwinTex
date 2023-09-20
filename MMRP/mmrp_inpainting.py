from PIL import Image
import os
import cv2
import yaml
import numpy as np
import argparse
import CommandRunner as cr
import shutil

mod_version = 46365
temp_dir=''
zero_mask_pth = ''
inpainting_path=''

cluster_line_file=''
k = 5

def center_crop(image, width, height):
    imag = Image.open(image)
    ori_width, ori_height = imag.size

    left = (ori_width - width) / 2
    top = (ori_height - height) / 2
    right = (ori_width + width) / 2
    bottom = (ori_height + height) / 2

    return imag.crop((left, top, right, bottom))

def rotate_bound(image, angle, borderColor=(255, 255, 255)):
    # grab the dimensions of the image and then determine the center
    im = cv2.imread(image)
    (h, w) = im.shape[:2]  
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(im, M, (nW, nH), borderValue=borderColor)


def rotate_pca(line_arr):
    xaxis = []
    yaxis = []
    xaxis = [line_arr[i][0] for i in range(len(line_arr))]
    xaxis.extend([line_arr[i][2] for i in range(len(line_arr))])
    yaxis = [line_arr[i][1] for i in range(len(line_arr))]
    yaxis.extend([line_arr[i][3] for i in range(len(line_arr))])
    data = np.stack([xaxis, yaxis]).T
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data, np.array([]))

    theta = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    theta2 = np.arctan2(eigenvectors[1, 1], eigenvectors[1, 0]) * 180 / np.pi
    angle_rotate = (theta if theta < theta2 else theta2)

    return angle_rotate


def extract_bb_mask(mask_img, img_name):
    mask_img = np.array(mask_img)
    mask_img_inv = 1 - mask_img
    mask_img_inv = Image.fromarray(mask_img_inv * 255.0)
    mask_img_inv = mask_img_inv.convert('L')
    mask_img_inv_path = temp_dir + '\\' + img_name + '_mask_inv.png'
    mask_img_inv.save(mask_img_inv_path)
    mask_ing_inv = Image.open(mask_img_inv_path)
    return mask_ing_inv.getbbox()


def automask(mask_pth):
    mask = Image.open(mask_pth).convert('1')
    img_name = mask_pth.split('\\')[-2]
    img_name = img_name.split('_')[0]
    print(extract_bb_mask(mask, img_name))
    (left, top, right, bottom) = extract_bb_mask(mask, img_name)
    width_split = (bottom - top) // 3

    mask1_img = mask.copy()
    mask2_img = mask.copy()
    mask3_img = mask.copy()

    mask1_datas = np.array(mask1_img)
    for y in range(top + width_split, 512):
        for x in range(0, 512):
            mask1_datas[y, x] = True

    mask2_datas = np.array(mask2_img)
    for x in range(0, 512):
        for y in range(0, top + width_split):
            mask2_datas[y, x] = True
        for y in range(bottom - width_split, 512):
            mask2_datas[y, x] = True

    mask3_datas = np.array(mask3_img)
    for x in range(0, 512):
        for y in range(0, bottom - width_split):
            mask3_datas[y, x] = True

    sketch_output_path1 = inpainting_path + '\\datasets\\gt_keep_masks\\' + img_name + '_twintex' +str(mod_version)+'_sketch1'
    sketch_output_path2 = inpainting_path + '\\datasets\\gt_keep_masks\\' + img_name + '_twintex' +str(mod_version)+'_sketch2'
    sketch_output_path3 = inpainting_path + '\\datasets\\gt_keep_masks\\' + img_name + '_twintex' +str(mod_version)+'_sketch3'
    if os.path.exists(sketch_output_path1) is False: os.makedirs(sketch_output_path1)
    if os.path.exists(sketch_output_path2) is False: os.makedirs(sketch_output_path2)
    if os.path.exists(sketch_output_path3) is False: os.makedirs(sketch_output_path3)

    mask1_output_path = sketch_output_path1 + '\\000000.png'
    mask1_output_im = Image.fromarray(mask1_datas).save(mask1_output_path)
    mask2_output_path = sketch_output_path2 + '\\000000.png'
    mask2_output_im = Image.fromarray(mask2_datas).save(mask2_output_path)
    mask3_output_path = sketch_output_path3 + '\\000000.png'
    mask3_output_im = Image.fromarray(mask3_datas).save(mask3_output_path)

    return sketch_output_path1, sketch_output_path2, sketch_output_path3


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--img_name',type=str, default=None, help='img_name')
    parse.add_argument('--known_area', type=float, default=None, help='known_area')
    parse.add_argument('--project_dir', type=str, default=None, help='project_dir')
    parse.add_argument('--inpainting_dir', type=str, default=None, help='inpainting_dir')
    parse.add_argument('--result_dir', type=str, default=None, help='result_dir')
    args=vars(parse.parse_args()) 
    
    project_dir = args.get('project_dir')
    inpainting_dir = args.get('inpainting_dir')
    #cluster line path
    cluster_line_file= args.get('cluster_line_file')
    #result path
    result_path = args.get('result_dir') 
    
    input_img_dir = inpainting_dir+'\\input'
    temp_dir = inpainting_dir + '\\temp'

    input_name = args.get('img_name') 
    input_img_path = input_img_dir + "\\" + input_name + ".png"

    if os.path.exists(input_img_path) is True:
        print(f"Inpaint {input_name}")
        orin_img_pth =  input_img_dir + "\\" + input_name + '_orin.png'
        orin_img = Image.open(orin_img_pth)
        orin_width,orin_height = orin_img.size

        crop_img = cv2.imread(input_img_path)

        mask_after_dilate = temp_dir + '\\' + input_name + '_mask_dilate_' + str(k) + '.png'  # mask for filtering in-the-boarder lines
        cluster_line_file = inpainting_dir +"\\cluster_line\\" + input_name + "_cluster_lines.txt"

        # rotate img and mask with p direction
        if os.path.exists(cluster_line_file):
            line_cordi = np.loadtxt(cluster_line_file)
            angle_rotate = rotate_pca(line_cordi)

            im = rotate_bound(input_img_path, angle_rotate, (0, 0, 0))
            mask = rotate_bound(mask_after_dilate, angle_rotate, (0, 0, 0))

            im_rotated_pth = temp_dir + '\\' + input_name + '_rotated.png'
            mask_rotated_pth = temp_dir + '\\' + input_name + '_mask_dilate20_rotated.png'

            cv2.imwrite(im_rotated_pth, im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(mask_rotated_pth, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            print(f'rotate finish, img in：{im_rotated_pth}')
            print(f'rotate finish, mask in：{mask_rotated_pth}')

            input_img_pth = im_rotated_pth
            mask_after_dilate = mask_rotated_pth
        else:
            input_img_pth = temp_dir + '\\'+input_name+'_rm_black_bdry.png'

    
        known_area = args.get('known_area')
        if known_area < 0.6:
            input_img_512_pth = temp_dir + '\\' + input_name + '_512.png'
        else:
            input_img = Image.open(input_img_pth).convert('RGB')
            input_img_512 = input_img.resize((512, 512))
            input_img_512_pth = temp_dir+'\\'+input_name+'_512.png'
            input_img_512.save(input_img_512_pth)


        mask_img = Image.open(mask_after_dilate).convert('RGB')
        mask_img_512 = mask_img.resize((512, 512))
        inpaint_mask_path = inpainting_dir + '\\datasets\\gt_keep_masks\\' + input_name + '_twintex' + str(mod_version)
        if os.path.exists(inpaint_mask_path) is False: os.makedirs(inpaint_mask_path)
        print(f'extract mask for inpainting，mask in：{inpaint_mask_path}')

        mask_img_512.save(inpaint_mask_path + '\\000000.png')

        # create gts dir and log dir
        gts_pth = inpainting_dir +'\\datasets\\gts\\' + input_name
        if os.path.exists(gts_pth) is False: os.makedirs(gts_pth)

        img = Image.open(input_img_512_pth).convert('RGB')
        img.save(inpainting_dir +'\\datasets\\gts\\' + input_name + '\\000000.png')
        print('gts dir created')
        print(f'gts in ：{gts_pth}')

        log_pth = inpainting_dir + '\\log\\' + input_name
        if os.path.exists(log_pth) is False:  os.makedirs(log_pth)
        log_inpainted = log_pth + '\\inpainted'
        if os.path.exists(log_pth + '\\inpainted') is False: os.makedirs(log_inpainted)
        log_gts_masked = log_pth + '\\gt_masked'
        if os.path.exists(log_pth + '\\gt_masked') is False: os.makedirs(log_gts_masked)
        log_gts =  log_pth + '\\gt'
        if os.path.exists(log_pth + '\\gt') is False: os.makedirs(log_gts)
        log_gt_keep_masks = log_pth + '\\gt_keep_mask'
        if os.path.exists(log_pth + '\\gt_keep_mask') is False: os.makedirs(log_gt_keep_masks)

        # print('log dir created')
        # print(f'log dir in：{log_pth}')

        # auto mask+gt_keep_masks dir
        mask1_output_pth, mask2_output_pth, mask3_output_pth = automask(inpaint_mask_path + '\\000000.png')
        print('mask auto mask finish')
        print(f'in：{mask1_output_pth}, {mask2_output_pth}, {mask3_output_pth}\n')

        # area that no need to inpaint
        if len(zero_mask_pth) != 0:
            zero_mask = Image.open(zero_mask_pth).convert('1')
            zero_mask_output_pth = inpainting_dir + '\\data\\datasets\\gt_keep_masks\\' + input_name + '_twintex' + str(
                mod_version) + '_filter_mask'
            if os.path.exists(zero_mask_output_pth) is False: os.makedirs(zero_mask_output_pth)
            zero_mask.save(zero_mask_output_pth + '\\000000.png')
        else:
            # white_mask
            white_mask = Image.open(project_dir + '\\white_mask.png').convert('1')
            white_mask_output_pth = inpainting_dir + '\\data\\datasets\\gt_keep_masks\\' + input_name + '_twintex' + str(
                mod_version) + '_filter_mask'
            if os.path.exists(white_mask_output_pth) is False: os.makedirs(white_mask_output_pth)
            white_mask.save(white_mask_output_pth + '\\000000.png')

        # create yaml
        example_yaml_pth = project_dir + '\\confs\\example_twintex46365.yml'
        new_yaml_pth = inpainting_dir + '\\confs'
        if os.path.exists(new_yaml_pth) is False: os.makedirs(new_yaml_pth)
        new_yaml_pth+='\\' + input_name + '_twintex46365.yml'
        with open(example_yaml_pth, 'r') as f:
            ex_yl = yaml.safe_load(f)
        new_yl = ex_yl
        new_yl['name'] = input_name #+ '_twintex45365'
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['gt_path'] = gts_pth
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['mask_path'] = mask1_output_pth
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['mask2_path'] = mask2_output_pth
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['mask3_path'] = mask3_output_pth
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['inpaint_mask_path'] = inpaint_mask_path
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['filter_mask_path'] = white_mask_output_pth
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['paths']['srs'] = log_pth + '\\inpainted'
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['paths']['lrs'] = log_pth + '\\gt_masked'
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['paths']['gts'] = log_pth + '\\gt'
        new_yl['data']['eval']['lama_inet256_thick_n100_test']['paths']['gt_keep_masks'] = log_pth + '\\gt_keep_mask'
        #data
        new_yl['known_area'] = known_area
        new_yl['model_path'] = project_dir+'\\data\\pretrained\\ema_0.9999_151161.pt'
        new_yl['input_path'] = input_img_dir
        new_yl['temp_pth'] =  temp_dir
        new_yl['result_path'] = inpainting_dir+'\\result'
        if os.path.exists(new_yl['result_path']) is False: os.makedirs(new_yl['result_path'])

        with open(new_yaml_pth, 'w', encoding='utf-8') as f2:
            yaml.dump(new_yl, f2)
        

        inpainting_path = project_dir + "\\inpaint.py"
        # command list
        run_command = 'python '+ inpainting_path +' --conf_path ' + new_yaml_pth
        cmd = cr.CommandRunner()
        cmd.run_cmd(run_command)

        im_inpainted_path = inpainting_dir + '\\result\\'+ input_name + '.png'
        im_inpainted_rotated_path = result_path + '\\' + input_name + '_inpainted.png'
        # if len(cluster_line_file) != 0:
        #     im_inpainted_rotated_path = result_path + '\\' + input_name + '_inpainted.png'
        #     im_inpainted_rotated = rotate_bound(im_inpainted_path, (-1) * angle_rotate, (0, 0, 0))
        #     cv2.imwrite(im_inpainted_rotated_path, im_inpainted_rotated, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        #     im3 = center_crop(im_inpainted_rotated_path, orin_width, orin_height)
        #     im3.save(im_inpainted_rotated_path)
        # else:
        #     shutil.copy(im_inpainted_path,im_inpainted_rotated_path)

        shutil.copy(im_inpainted_path,im_inpainted_rotated_path)

        print(f'result: {im_inpainted_rotated_path}\n')
    else:
        print(f"{input_img_path} is not Exist" )