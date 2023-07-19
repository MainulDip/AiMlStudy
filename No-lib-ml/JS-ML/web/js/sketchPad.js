class SketchPad {
    constructor(container, size = 400) {
        this.canvas = document.createElement("canvas")
        this.canvas.width = size
        this.canvas.height = size
        this.canvas.style = `background-color: white; box-shadow: 0px 0px 10px 2px black; `

        container.appendChild(this.canvas)

        // line break and undo button
        const lineBreak = document.createElement("br")
        container.appendChild(lineBreak)
        this.undoBtn = document.createElement("button")
        this.undoBtn.innerHTML = "UNDO"
        container.appendChild(this.undoBtn)


        // getting canvas 2d context
        this.ctx = this.canvas.getContext("2d")

        // set some class property and populate them from those listeners
        this.paths = []
        this.isDrawing = false

        // prevent the undo button from being activated
        this.#reDraw()

        // call an event listener
        this.#addEventListeners();

        
        
    }

    // hash "#" denotes it as private method
    #addEventListeners() {
        // mousedown event
        this.canvas.onmousedown = (evt) => {
            const mouse = this.#getMouse(evt)

            // console.log(mouse)

            // update class props
            // path = [[x,y],[x,y],[x,y],...] and mouse = [x,y]
            this.paths.push([mouse])
            this.isDrawing = true
        }

        // mousemove event, only fire when drawing
        this.canvas.onmousemove = (evt) => {
            if (this.isDrawing) {
                const mouse = this.#getMouse(evt)

                // update class props with array of x & y mouse coordinate | [[x,y],[x,y],[x,y],...]
                // get the last path position
                const lastPath = this.paths[this.paths.length - 1]
                lastPath.push(mouse)
                console.log(this.path)

                // call the drawing method as redraw
                this.#reDraw()
            }
        }

        this.canvas.onmouseup = () => {
            this.isDrawing = false
        }

        // touch events
        this.canvas.ontouchstart = (evt) => {
            const loc = evt.touches[0]
            this.canvas.onmousedown(loc)
        }

        this.canvas.ontouchmove = (evt) => {
            const loc = evt.touches[0]
            this.canvas.onmousemove(loc)
        }

        this.canvas.ontouchend = () => {
            this.canvas.onmouseup(loc)
        }

        // undo btn
        this.undoBtn.onclick = (evt) => {
            this.paths.pop()
            this.#reDraw()
        }
    }

    #getMouse = (evt) => {
        const rect = this.canvas.getBoundingClientRect();
        // returning back array x and y (2d) position of mouse while on pressed
        return [
            Math.round(evt.clientX - rect.left),
            Math.round(evt.clientY - rect.top)
        ]
    }

    // Drawing Method
    #reDraw() {
        this.ctx.clearRect(0,0, this.canvas.width, this.canvas.height)
        
        // call the draw util object to actually draw on the canvas
        draw.paths(this.ctx, this.paths)

        if(this.paths.length > 0) {
            this.undoBtn.disabled = false
        } else {
            this.undoBtn.disabled = true
        }
    }
}