# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['postProcessing.py'],
             pathex=['C:\\Users\\Shawlab\\Desktop\\HerbASAP'],
             # have to show it how to find the custom pyzbar dlls
             binaries=[('C:/Users/Shawlab/Desktop/HerbASAP/libs/deps/pyzbar/libiconv.dll', '.'),
                       ('C:/Users/Shawlab/Desktop/HerbASAP/libs/deps/pyzbar/libiconv-2.dll', '.'),
                       ('C:/Users/Shawlab/Desktop/HerbASAP/libs/deps/pyzbar/libzbar-32.dll', '.'),
                       ('C:/Users/Shawlab/Desktop/HerbASAP/libs/deps/pyzbar/libzbar-64.dll', '.'),
                       (HOMEPATH + '\\pylibdmtx\\libdmtx-64.dll', '.')
                       ],
             # for Qt binary misplacement issue see below
             # https://github.com/pyinstaller/pyinstaller/issues/4293
             datas=[(HOMEPATH + '\\PyQt5\\Qt\\bin\*', 'PyQt5\\Qt\\bin'),
		    ('C:/Users/Shawlab/Desktop/HerbASAP/libs/models/*', 'libs/models'),
		    ('C:/Users/Shawlab/Desktop/HerbASAP/libs/models/f_rcnn_lcc.hdf5', 'libs/models'),
		    ('C:/Users/Shawlab/Desktop/HerbASAP/ui/styles/darkorange/*', 'ui/styles/darkorange'),
		    ('C:/Users/Shawlab/Desktop/HerbASAP/docs/icon_a.ico', 'docs/icon_a.ico')],
             # this tensorflow hidden import takes care lib missed by pyinstaller
             hiddenimports=["tensorflow.lite.python.interpreter_wrapper.tensorflow_wrap_interpreter_wrapper"],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4','matplotlib','wx','IPython','tkinter','tk'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
	  a.zipfiles,
	  a.datas,
          #exclude_binaries=True,
          name='HerbASAP',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True,
	  runtime_tmpdir=None,
	  icon='C:/Users/Shawlab/Desktop/HerbASAP/docs/icon_a.ico')

#coll = COLLECT(exe,
#               a.binaries,
#               a.zipfiles,
#               a.datas,
#               strip=False,
#               upx=True,
#               upx_exclude=[],
#               name='HerbASAP')
