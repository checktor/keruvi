const express = require('express');
const router = require('./router.js');

// Create app.
const app = express();

// Serve static files from 'static' directory.
app.use(express.static('static'));

// Read body from 'Content-Type: application/json' requests.
app.use(express.json());

// Serve custom routes.
app.use('/', router);

const PORT = process.env.KERUVI_PORT || 3000;

app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}.`);
});