# coding: utf-8
import unittest
from predict_vision import calc_diff


class InferenceModelDiffTest(unittest.TestCase):
    # pass
    def test_vgg(self):
        from models.vision import tlx_vgg, pd_vgg

        image_file = "images/dog.jpeg"
        model_tlx = tlx_vgg.vgg11(pretrained=False)
        model_pd = pd_vgg.vgg11(pretrained=False)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_vgg.vgg13(pretrained=False)
        model_pd = pd_vgg.vgg13(pretrained=False)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_vgg.vgg16(pretrained=True)
        model_pd = pd_vgg.vgg16(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_vgg.vgg19(pretrained=True)
        model_pd = pd_vgg.vgg19(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    # small diff
    def test_alexnet(self):
        from models.vision import tlx_alexnet, pd_alexnet

        image_file = "images/dog.jpeg"
        model_tlx = tlx_alexnet.alexnet(pretrained=True)
        model_pd = pd_alexnet.alexnet(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    def test_googlenet(self):
        from models.vision import tlx_googlenet, pd_googlenet

        image_file = "images/dog.jpeg"
        model_tlx = tlx_googlenet.googlenet(pretrained=True)
        model_pd = pd_googlenet.googlenet(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    def test_squeezenet(self):
        from models.vision import tlx_squeezenet, pd_squeezenet

        image_file = "images/dog.jpeg"
        model_tlx = tlx_squeezenet.squeezenet1_0(pretrained=True)
        model_pd = pd_squeezenet.squeezenet1_0(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_squeezenet.squeezenet1_1(pretrained=True)
        model_pd = pd_squeezenet.squeezenet1_1(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    # pass
    def test_resnet(self):
        from models.vision import tlx_resnet, pd_resnet

        image_file = "images/dog.jpeg"
        model_tlx = tlx_resnet.resnet18(pretrained=True)
        model_pd = pd_resnet.resnet18(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_resnet.resnet34(pretrained=True)
        model_pd = pd_resnet.resnet34(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_resnet.resnet50(pretrained=True)
        model_pd = pd_resnet.resnet50(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_resnet.resnet101(pretrained=True)
        model_pd = pd_resnet.resnet101(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_resnet.resnet152(pretrained=True)
        model_pd = pd_resnet.resnet152(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    def test_densenet(self):
        from models.vision import tlx_densenet, pd_densenet

        image_file = "images/dog.jpeg"
        model_tlx = tlx_densenet.densenet121(pretrained=True)
        model_pd = pd_densenet.densenet121(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_densenet.densenet161(pretrained=True)
        model_pd = pd_densenet.densenet161(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_densenet.densenet169(pretrained=True)
        model_pd = pd_densenet.densenet169(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_densenet.densenet201(pretrained=True)
        model_pd = pd_densenet.densenet201(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_densenet.densenet264(pretrained=True)
        model_pd = pd_densenet.densenet264(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    # not pass
    def test_mobilenet(self):
        from models.vision import tlx_mobilenetv1, tlx_mobilenetv2, tlx_mobilenetv3
        from models.vision import pd_mobilenetv1, pd_mobilenetv2, pd_mobilenetv3

        image_file = "images/dog.jpeg"
        model_tlx = tlx_mobilenetv1.mobilenet_v1(pretrained=True)
        model_pd = pd_mobilenetv1.mobilenet_v1(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_mobilenetv2.mobilenet_v2(pretrained=True)
        model_pd = pd_mobilenetv2.mobilenet_v2(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        # model_tlx = tlx_mobilenetv3.mobilenet_v3_small(pretrained=True)
        # model_pd = tlx_mobilenetv3.mobilenet_v3_small(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file)
        #
        # model_tlx = tlx_mobilenetv3.mobilenet_v3_large(pretrained=True)
        # model_pd = pd_mobilenetv3.mobilenet_v3_large(pretrained=True)
        # calc_diff(model_tlx, model_pd, image_file)

    def test_inception(self):
        from models.vision import tlx_inceptionv3

        image_file = "images/dog.jpeg"
        model_tlx = tlx_inceptionv3.inception_v3(pretrained=True)
        model_pd = tlx_inceptionv3.inception_v3(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    # pass
    def test_shufflenet(self):
        from models.vision import tlx_shufflenetv2, pd_shufflenetv2

        image_file = "images/dog.jpeg"
        model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_33(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x0_33(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_shufflenetv2.shufflenet_v2_x1_0(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x1_0(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_shufflenetv2.shufflenet_v2_x1_5(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x1_5(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_shufflenetv2.shufflenet_v2_x2_0(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x2_0(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        model_pd = pd_shufflenetv2.shufflenet_v2_x0_25(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    def test_darknet(self):
        from models.vision import tlx_darknet53, pd_darknet53

        image_file = "images/dog.jpeg"
        model_tlx = tlx_darknet53.darknet53(pretrained=True)
        model_pd = pd_darknet53.darknet53(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

    def test_rednet(self):
        from models.vision import tlx_rednet, pd_rednet

        image_file = "images/dog.jpeg"
        model_tlx = tlx_rednet.RedNet50(pretrained=True)
        model_pd = pd_rednet.RedNet50(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)

        model_tlx = tlx_rednet.RedNet101(pretrained=True)
        model_pd = pd_rednet.RedNet101(pretrained=True)
        calc_diff(model_tlx, model_pd, image_file)


if __name__ == '__main__':
    unittest.main()
