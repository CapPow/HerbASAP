#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    This program is free software; you can redistribute it and/or
#    modify it under the terms of the GNU General Public License
#    as published by the Free Software Foundation; either version 3
#    of the License, or (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""

    AYUP (as of yet unnamed program) performs post processing steps on raw 
    format images of natural history specimens. Specifically designed for 
    Herbarium sheet images.

"""
# imports here

class ccRead():
    def __init__(self, parent=None, *args):
        super(ccRead, self).__init__()
        self.parent = parent
        
    def testFeature(self, img):
        """Returns bool condition, if this module functions on a test input."""

        try:
            # Check if the funcions return expected dtypes
            #if isinstance(self.retrieveBcMatches(img), list):
                return True
            else:
                return False
        except:
            # some unknown error, assume test failed
            return False
