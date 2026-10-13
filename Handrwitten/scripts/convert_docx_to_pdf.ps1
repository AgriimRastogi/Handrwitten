$docxPath = "c:\Users\basub\OneDrive\Desktop\AR-Handwriting Synthesis\SRS_v1.0_22515500004_HandwritingSynthesis.docx"
$pdfPath = "c:\Users\basub\OneDrive\Desktop\AR-Handwriting Synthesis\SRS_v1.0_22515500004_HandwritingSynthesis.pdf"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docxPath)
    $doc.SaveAs([ref]$pdfPath, [ref]17)
    $doc.Close()
    Write-Output "PDF generated successfully at $pdfPath"
} catch {
    Write-Error $_.Exception.Message
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
