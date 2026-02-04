class ChemToolError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 400,
        detail: dict | None = None
    ):
        self.code = code                  
        self.message = message           
        self.http_status = http_status    
        self.detail = detail or {}        
        super().__init__(message)
