const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
    transpileDependencies: [
        '@mywai/3d-viewer',
        '@mkkellogg/gaussian-splats-3d'
    ],
    configureWebpack: {
        resolve: {
            alias: {
                // Three.js addon path aliases (matching IOTA_DApp pattern)
                'three/addons/loaders/GLTFLoader.js': 'three/examples/jsm/loaders/GLTFLoader.js',
                'three/addons/loaders/DRACOLoader.js': 'three/examples/jsm/loaders/DRACOLoader.js',
                'three/addons/loaders/FBXLoader.js': 'three/examples/jsm/loaders/FBXLoader.js',
                'three/addons/loaders/OBJLoader.js': 'three/examples/jsm/loaders/OBJLoader.js',
                'three/addons/loaders/MTLLoader.js': 'three/examples/jsm/loaders/MTLLoader.js',
                'three/addons/loaders/PLYLoader.js': 'three/examples/jsm/loaders/PLYLoader.js',
                'three/addons/loaders/STLLoader.js': 'three/examples/jsm/loaders/STLLoader.js',
                'three/addons/loaders/ColladaLoader.js': 'three/examples/jsm/loaders/ColladaLoader.js',
                'three/addons/loaders/TDSLoader.js': 'three/examples/jsm/loaders/TDSLoader.js',
                'three/addons/controls/OrbitControls.js': 'three/examples/jsm/controls/OrbitControls.js',
            }
        }
    }
})
