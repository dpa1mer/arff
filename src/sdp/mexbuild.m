function mexbuild(pathToTbbInclude, pathToMosek)

if ismac
    platform = 'osx64x86';
elseif isunix
    platform = 'linux64x86';
end

mosekIncludeDir = fullfile(pathToMosek, 'tools/platform', platform, 'h');
mosekLibDir = fullfile(pathToMosek, 'tools/platform', platform, 'bin');

ldflags = '';
if ~ismac && isunix
    ldflags = ['LDFLAGS=$LDFLAGS -Wl,-rpath=' mosekLibDir];
end

mex('MultiSdp.cpp', ...
    '-cxx', '-O', '-g', ['-I' pathToTbbInclude], ...
    ['-I' mosekIncludeDir], ['-L' mosekLibDir], ...
    '-lmosek64', '-lfusion64', '-ltbb', ldflags);

if ismac
    system(['install_name_tool -change libmosek64.9.0.dylib ' fullfile(mosekLibDir, 'libmosek64.9.0.dylib') ' MultiSdp.mexmaci64']);
    system(['install_name_tool -change libfusion64.9.0.dylib ' fullfile(mosekLibDir, 'libfusion64.9.0.dylib') ' MultiSdp.mexmaci64']);
end

end