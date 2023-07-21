const draw = {}

// to draw single line/stroke/path by connecting dots/position of mouse (x,y)
// ctx is the 2d canvas context and path is 
draw.path = (ctx, path, color="black") => { // path = [[x,y],[x,y],[x,y],...]
    ctx.strokeStyle = color
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(...path[0]) // set the pointer to path[0] as starting position

    // Now start drawing by connecting the positions (x,y) by looping after pointing the starting point before this by moveTo
    // path = [[x,y],[x,y],[x,y],...]
    for (let i = 1; i < path.length; i++) {
        ctx.lineTo(...path[i]) // spreading [x,y] array into x and y int
    }

    // make line joining and ending round corner
    ctx.lineCap = "round"
    ctx.lineJoin = "round"
    
    ctx.stroke();
}

// draw multiple paths, called form sketchpad.js
/* 
& paths = [[[x,y],[x,y],...],
             [[x,y],[x,y],...], 
             [[x,y],[x,y],...], 
             ...]
*/
draw.paths = (ctx, paths, color="black") => {
    for (const path of paths) {
        // path = [[x,y],[x,y],[x,y],...]
        draw.path(ctx, path, color)
    }
}


// preventing browsers as module.exports is a node.js specific feature
if(typeof module!=='undefined'){
    module.exports = draw
 }