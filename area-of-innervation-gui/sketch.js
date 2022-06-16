let pts;
let numCircles;
const images = [];
let currImage;

function preload() {
  currImage = parseInt(window.location.search.substring(1));
  currImage = (!currImage || currImage < 1) ? 1 : (currImage > 22) ? 22 : currImage;
  for (let i = 0; i < 22; i++) {
    images.push(loadImage("images/"+(i+1)+".png"));
  }
}

function setup() {
  canv = createCanvas(512, 512);
  canv.parent("canv")
  addAreaBtn = createButton("Add Area")
  addAreaBtn.mousePressed(addArea)
  addAreaBtn.parent("canv")
  clearBtn = createButton("Clear")
  clearBtn.mousePressed(clearPts)
  clearBtn.parent("canv")
  prevBtn = createButton("Previous")
  prevBtn.mousePressed(prev)
  prevBtn.parent("canv")
  nextBtn = createButton("Next")
  nextBtn.mousePressed(next)
  nextBtn.parent("canv")
  clearPts()
}

function clearPts() {
  pts = [[]];
  numCircles = 0;
  const out = document.getElementById("points")
  out.innerHTML = "Image " + currImage
}

function next() {
  currImage = (currImage % 22) + 1
  clearPts()
}

function prev() {
  currImage = ((currImage+20) % 22) + 1
  clearPts()
}

function addArea() {
  if (pts[numCircles].length > 1) {
    numCircles += 1
    pts.push([])
  }
  const out = document.getElementById("points")
  out.innerHTML = "Image " + currImage + "<br><br>" + pts.map((l,i) =>
    l.length > 1 ? "Area " + (i+1) + "<br>" + l.map(pt => pt.x+" "+pt.y).join("<br>") : ""
  ).join("<br><br>")
}

function inBounds() {
  return mouseX < width && mouseY < height && mouseX >= 0 && mouseY >= 0
}

function mousePressed() {
  if (inBounds()) {
    pts[numCircles] = [new Point(round(mouseX),round(mouseY))];
  }
}

function mouseDragged() {
  if (inBounds()) {
    const len = pts[numCircles].length
    const newpt = new Point(round(mouseX),round(mouseY))
    if (len && newpt.dist(pts[numCircles][len-1]) < 10) {
      return
    }
    pts[numCircles].push(new Point(round(mouseX),round(mouseY)));
  }
}

function drawCircle(l) {
  push();
  strokeWeight(1);
  noFill();
  beginShape();
  for (let i = 0; i < l.length; i++) {
    vertex(l[i].x, l[i].y);
  }
  endShape(CLOSE);
  pop();
}

function draw() {
  image(images[currImage-1],0,0)
  stroke(255);
  pts.forEach((l,i) => {
    if (i == numCircles) {
      stroke(140);
    }
    drawCircle(l);
  })
}
