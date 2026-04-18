// Newton School Online TypeScript compiler to run TypeScript online.
// Write TypeScript code in this online editor and run it.

class File_Node{
    private name:string
    private size:number

    constructor(a:string,b:number){
        this.size=b
        this.name=a
    }
    getSize():number{
        return 
    }
}

class Folder_node{
    private files:File_Node[]
    private folders:Folder_node[]
    private name:string
    constructor(name:string){
        this.name=name
        this.files=[]
        this.folders=[]
    }
    add_file(file:File_Node):void{}
    add_folder(folder:Folder_node):void{
        this.folders.push(folder)
    }
    getSize():number{
        let total:number=0
        for(let i of this.files){
            total+=i.getSize()
        }
        for( let i of this.folders){
            total += i.getSize()
        }
        return total
    }
}

let root: Folder_node=new Folder_node('root')
let folder1: Folder_node=new Folder_node('folder1')
let folder2: Folder_node=new Folder_node('folder2')
let file1: Folder_node=new Folder_node('file1',5)
let file2: Folder_node=new Folder_node('file2',3)

notify(message: string): void {     
    
let file3: Folder_node=new Folder_node('file3',7)


folder1.add_file(file1)
folder2.add_file(file2)
root.add_folder(folder1)
root.add_folder(folder2)
root.add_file(file3)
console.log(root.getSize())