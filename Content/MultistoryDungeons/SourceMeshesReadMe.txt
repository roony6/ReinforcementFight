"SourceMeshes.rar" contains source .fbx files.

If you would like to modify these files and intergare them into your project,
please unpack "SourceMeshes.rar" to Content > MultistoryDungeons

Resulting folder structure should look like this:

>Content
  >MultistoryDungeons
    >Source_FBX
    >Source_FBX_Props

In the Editor, please make sure to click "Don't Import" when the little window
pops up, telling you that UE4 detected source files changes.
Otherwise this will break statick meshes (reported many times for UE4.14)

When you modify and import a mesh, please pay close attention to its import
setting. For example, if a source file contains custom collision, make sure
to uncheck "auto generate collision" ect.