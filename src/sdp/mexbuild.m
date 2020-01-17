function mexbuild(pathToTbbInclude, pathToMosek, pathToTbbLink)

if ismac
    platform = 'osx64x86';
elseif isunix
    platform = 'linux64x86';
elseif ispc
    platform = 'win64x86';
end

mosekIncludeDir = fullfile(pathToMosek, 'tools/platform', platform, 'h');
mosekLibDir = fullfile(pathToMosek, 'tools/platform', platform, 'bin');

ldflags = '';
if ~ismac && isunix
    ldflags = ['LDFLAGS=$LDFLAGS -Wl,-rpath=' mosekLibDir];
end

mex('MultiSdp.cpp', ...
    '-cxx', 'COMPFLAGS="\$COMPFLAGS -MT"', '-O', '-g', ['-I' pathToTbbInclude], ...
    ['-I' mosekIncludeDir], ['-L' mosekLibDir], ['-L' pathToTbbLink], ...
    '-lmosek64_9_0', '-lfusion64_9_0', ldflags);

if ismac
    system(['install_name_tool -change libmosek64.9.0.dylib ' fullfile(mosekLibDir, 'libmosek64.9.0.dylib') ' MultiSdp.mexmaci64']);
    system(['install_name_tool -change libfusion64.9.0.dylib ' fullfile(mosekLibDir, 'libfusion64.9.0.dylib') ' MultiSdp.mexmaci64']);
end

end