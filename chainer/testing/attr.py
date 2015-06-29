from nose.plugins import attrib

gpu = attrib.attr('gpu')
cudnn = attrib.attr('gpu', 'cudnn')
