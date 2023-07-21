const draw = require('../common/draw.js')
const constants = require('../common/constants.js')
const utils = require('../common/utils.js')

const {createCanvas} = require('canvas')

const canvas = createCanvas(400, 400)
const ctx = canvas.getContext("2d")



const fs = require('fs')

// reading all the file in raw directory
const fileNames = fs.readdirSync(constants.RAW_DIR)
// the collected data will be re-arranged in this array and will create a new json file based on it
const samples = []
let id = 1

// looping over each file inside raw directory
fileNames.forEach( fn => { // fn for filename
  // getting the content/reading each file by name
  const content = fs.readFileSync( constants.RAW_DIR + "/" + fn )
  
  // destructuring parsed data as these
  const {session, student, drawings} = JSON.parse(content)

  // populating samples with id for each image label (ie: bicycle, tree, pencil etc), so we can play with those data better later
  // so there will 8 different label
  for (let label in drawings) { // getting key as label
    samples.push ({
      id,
      label,
      student_name : student,
      student_id : session
    })

    // creating separate json file based on label of the path array, with filename prefixed with matching id 
    const paths = drawings[label]
    fs.writeFileSync(
      constants.JSON_DIR + "/" + id + ".json", JSON.stringify(paths)
    )

    // creating img file based on each paths
    generateImageFile(
      constants.IMG_DIR + "/" + id + ".png", paths
    )

    // show the image generation progress, each file contains 8 image
    utils.printProgress(id, fileNames.length * 8)
    id++ 
  } 
})

// creating a new file based on the sample array
fs.writeFileSync(constants.SAMPLES, JSON.stringify(samples))

// generation a separate samples for feeding web frontend to bypass CROS Error
fs.writeFileSync(constants.SAMPLES_JS, "const samples = " + JSON.stringify(samples))

function generateImageFile(outFile, paths) {
  ctx.clearRect(0,0, canvas.width, canvas.height)

  // image will be drawn on the ctx, so can grab later through canvas itself
  draw.paths(ctx, paths) 
  
  // converting into buffer
  const buffer = canvas.toBuffer("image/png") 

  fs.writeFileSync(outFile, buffer)
}

// preventing browsers as module.exports is a node.js specific feature
if(typeof module!=='undefined'){
   module.exports=constants;
}