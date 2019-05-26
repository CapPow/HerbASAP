"""

"""
import re
from pyzbar.pyzbar import decode

class bcRead():
    def __init__(self, parent=None, *args):
        super(bcRead, self).__init__()
        self.parent = parent
        self.compileRegexPattern()

    def compileRegexPattern(self):
        """ precompiles collection's regex pattern """
        prefix = self.parent.lineEdit_catalogNumberPrefix.text()
        digits = int(self.parent.spinBox_catalogDigits.value())
        # triple { required for combining regex and f strings
        collRegEx = fr'^({prefix}\d{{{digits}}})\D*'
        rePattern = re.compile(collRegEx)
        
        # set the compiled regex as a class attribute.
        self.rePattern = rePattern
        
        # This might need promoted to a user preference in mainwindow
        self.rotationList = [12, -12,  21, -21]

    def checkPattern(self, bcData):
        """ verifies if the bcData matches the compiled rePattern.
        Returns bool """

        if self.rePattern.match(bcData):
            return True
        else:
            return False
    
    def decodeBC(self, img):
        """given an openCV object (img), decodes BCs and returns those
        which match self.rePattern"""

        bcRawData = decode(img)
        bcStrings = [x.data.decode("utf-8") for x in bcRawData]
        bcData = [x for x in bcStrings if self.checkPattern(x)]  # get the barcode data
        return bcData

    def retrieveBcMatches(self, img):
        """ given na openCV object (img), retrieves the BC matches """

        bcData = self.decodeBC(img)
        if len(bcData) < 1:
            # if no barcodes are detected
            for i in self.rotationList:
                img2 = img.rotate((i), resample=Image.NEAREST, expand=True)
                bcData = self.decodeBC(img2)
                if len(bcData) > 0:
                    break
            #Give up on rotation, and handle null result
            return None

        elif len(bcData) == 1:
            # if only 1 barcode was detected:
            return bcData

        else:
            # if there are > 1 barcodes detected.
            # TODO if there are >1 bc, save the output under each valid bc number.
            # for now BREAK it
            breakPOINT

    def testFeature(self, img):
        """Returns bool condition, if this module functions on a test input."""

        try:
            if isinstance(self.retrieveBcMatches(img), list):
                return True
            else:
                return False
        except:
            # some unknown error, assume test failed
            return False

