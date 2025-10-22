document.getElementById("uploadForm").addEventListener("submit", function(event) {
    event.preventDefault();
    
    let formData = new FormData(this);
    fetch("/upload_detect", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        location.reload();
    })
    .catch(error => console.error("Error:", error));
});

// Fungsi Filter Data
document.getElementById("searchInput").addEventListener("keyup", function() {
    let filter = this.value.toLowerCase();
    let rows = document.querySelectorAll("#dataTable tr");

    rows.forEach(row => {
        let nama = row.cells[1].textContent.toLowerCase();
        let alamat = row.cells[2].textContent.toLowerCase();
        let tanggal = row.cells[3].textContent.toLowerCase();

        if (nama.includes(filter) || alamat.includes(filter) || tanggal.includes(filter)) {
            row.style.display = "";
        } else {
            row.style.display = "none";
        }
    });
});

// Fungsi Filter Data Tagihan
document.getElementById("searchInput").addEventListener("keyup", function() {
    let filter = this.value.toLowerCase();
    let rows = document.querySelectorAll("#tagihanTable tr");

    rows.forEach(row => {
        let nama = row.cells[1].textContent.toLowerCase();
        let alamat = row.cells[2].textContent.toLowerCase();

        if (nama.includes(filter) || alamat.includes(filter)) {
            row.style.display = "";
        } else {
            row.style.display = "none";
        }
    });
});


// Fungsi Export PDF
function exportToPDF(tableId, filename) {
    const { jsPDF } = window.jspdf;
    let doc = new jsPDF();

    let table = document.getElementById(tableId);
    let rows = table.getElementsByTagName("tr");

    let data = [];
    for (let i = 0; i < rows.length; i++) {
        let rowData = [];
        let cells = rows[i].getElementsByTagName("td");
        if (cells.length === 0) {
            cells = rows[i].getElementsByTagName("th"); // Ambil header jika ada
        }
        for (let j = 0; j < cells.length; j++) {
            rowData.push(cells[j].innerText);
        }
        data.push(rowData);
    }

    let startY = 20;
    doc.text(filename, 14, 10);
    doc.autoTable({
        head: [data[0]],
        body: data.slice(1),
        startY: startY
    });

    doc.save(`${filename}.pdf`);
}