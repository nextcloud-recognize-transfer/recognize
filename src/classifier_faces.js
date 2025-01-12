const path = require('path')
const fs = require('fs/promises')

let tf, Human, Jimp, wasm
let PUREJS = false
if (process.env.RECOGNIZE_PUREJS === 'true') {
	tf = require('@tensorflow/tfjs')
	wasm = require('@tensorflow/tfjs-backend-wasm')
	Human = require('@vladmandic/human/dist/human.node-wasm.js')
	Jimp = require('jimp')
	PUREJS = true
} else {
	try {
		if (process.env.RECOGNIZE_GPU === 'true') {
			tf = require('@tensorflow/tfjs-node-gpu')
			Human = require('@vladmandic/human/dist/human.node-gpu.js')
		} else {
			tf = require('@tensorflow/tfjs-node')
			Human = require('@vladmandic/human/dist/human.node.js')
		}
	} catch (e) {
		console.error(e)
		console.error('Trying js-only mode')
		tf = require('@tensorflow/tfjs')
		wasm = require('@tensorflow/tfjs-backend-wasm')
		Human = require('@vladmandic/human/dist/human.node-wasm.js')
		Jimp = require('jimp')
		PUREJS = true
	}
}
if (process.argv.length < 3) throw new Error('Incorrect arguments: node classifier_faces.js ...<IMAGE_FILES> | node classify.js -')

const config = {
	cacheSensitivity: 0.01,
	//modelBasePath: 'file://node_modules/@vladmandic/human/models/',
	modelBasePath: 'https://vladmandic.github.io/human-models/models/',
	backend: PUREJS ? 'wasm' : 'tensorflow',
	//wasmPath: 'file://node_modules/@tensorflow/tfjs-backend-wasm/dist/',
	wasmPath: `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tf.version_core}/dist/`,
	debug: false,
	async: false,
	softwareKernels: true,
	face: {
		enabled: true,
		detector: { rotation: false, maxDetected: 100 },
		iris: { enabled: true },
		description: { enabled: true },
		antispoof: { enabled: true },
		liveness: { enabled: true },
	},
	hand: { enabled: false },
	body: { enabled: false },
	object: { enabled: false },
	segmentation: { enabled: false },
	filter: { enabled: false },
}

/**
 *
 */
async function main() {
	const getStdin = (await import('get-stdin')).default
	let paths
	if (process.argv[2] === '-') {
		paths = (await getStdin()).split('\n')
	} else {
		paths = process.argv.slice(2)
	}

	Human.env.updateBackend();
	const human = new Human.Human(config)

	for (const path of paths) {
		try {
			let tensor
			if (PUREJS) {
				tensor = await createTensor(await Jimp.read(path), 3)
			} else {
				tensor = await tf.node.decodeImage(await fs.readFile(path), 3)
			}
			const results = await human.detect(tensor)
			tensor.dispose()
			const vectors = results.face
				.map(result => ({
					angle: result.rotation.angle,
					vector: result.embedding,
					x: result.boxRaw[0],
					y: result.boxRaw[1],
					height: result.boxRaw[3],
					width: result.boxRaw[2],
					score: result.score,
				}))
			console.log(JSON.stringify(vectors))
		} catch (e) {
			console.error(e)
			console.log('[]')
		}
	}
}

if (PUREJS) {
	wasm.setWasmPaths(config.wasmPath, true);
}
tf.setBackend(PUREJS ? 'wasm' : 'tensorflow')
	.then(() => main())
	.catch(e => {
		console.error(e)
	})

/**
 * @param image
 */
async function createTensor(image) {
	const NUM_OF_CHANNELS = 3
	const values = new Float32Array(image.bitmap.width * image.bitmap.height * NUM_OF_CHANNELS)
	let i = 0
	image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y) => {
		const pixel = Jimp.intToRGBA(image.getPixelColor(x, y))
		values[i * NUM_OF_CHANNELS + 0] = pixel.r
		values[i * NUM_OF_CHANNELS + 1] = pixel.g
		values[i * NUM_OF_CHANNELS + 2] = pixel.b
		i++
	})
	const outShape = [
		image.bitmap.height,
		image.bitmap.width,
		NUM_OF_CHANNELS,
	]
	const imageTensor = tf.tensor3d(values, outShape, 'float32')
	return imageTensor
}
