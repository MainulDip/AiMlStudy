class SketchPad {
    constructor(container, size = 400) {
        this.canvas = document.createElement("canvas")
        this.canvas.width = size
        this.canvas.height = size
        this.canvas.style = `background-color: white; box-shadow: 0px 0px 10px 2px black; `

        container.appendChild(this.canvas)
        this.ctx = this.canvas.getContext("2d")

        // call an event listener
        this.#addEventListeners();

        // set some class property and populate them from those listeners
        this.path = []
        this.isDrawing = false
    }

    // hash "#" denotes it as private method
    #addEventListeners() {
        // mousedown event
        this.canvas.onmousedown = (evt) => {
            const mouse = this.#getMouse(evt)

            // console.log(mouse)

            // update class props
            this.path = [mouse]
            this.isDrawing = true
        }

        // mousemove event, only fire when drawing
        this.canvas.onmousemove = (evt) => {
            if (this.isDrawing) {
                const mouse = this.#getMouse(evt)

                // update class props
                this.path.push(mouse)
                console.log(this.path.length)

                // call the drawing method as redraw
                this.#reDraw()
            }
        }

        this.canvas.onmouseup = () => {
            this.isDrawing = false
        }
    }

    #getMouse = (evt) => {
        const rect = this.canvas.getBoundingClientRect();
        return [
            Math.round(evt.clientX - rect.left),
            Math.round(evt.clientY - rect.top)
        ]
    }

    // Drawing Method
    #reDraw() {
        this.ctx.clearRect(0,0, this.canvas.width, this.canvas.height)
        
        // call the draw util object to actually draw on the canvas
        draw.path(this.ctx, this.path)
    }
}