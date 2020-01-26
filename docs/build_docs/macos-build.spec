# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['postProcessing.py'],
             pathex=['/Users/dakila/Documents/Git/HerbASAP/'],
             # have to show it how to find the custom pyzbar dlls
             binaries=[],
             # for Qt binary misplacement issue see below
             # https://github.com/pyinstaller/pyinstaller/issues/4293
             datas=[],
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
          [],
          exclude_binaries=True,
          name='HerbASAP',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )

app = BUNDLE(exe,
            info_plist={
               'NSRequiresAquaSystemAppearance': 'No'
            },
             name='HerbASAP.app',
             bundle_identifier=None,
             icon='icon_a.icns')


coll = COLLECT(exe,
              a.binaries,
              a.zipfiles,
              a.datas,
              strip=False,
              upx=True,
              upx_exclude=[],
              name='HerbASAP')
